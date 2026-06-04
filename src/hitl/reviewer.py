"""
Expert reviewer interface for HITL feedback collection
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExpertFeedback:
    """Container for expert review feedback"""
    item_id: str
    reviewer_id: str
    timestamp: float
    corrected_label: Optional[str] = None
    corrected_class_id: Optional[int] = None
    confidence_in_review: float = 1.0  # Expert's confidence in their review
    notes: str = ""
    flagged_for_escalation: bool = False
    additional_annotations: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'item_id': self.item_id,
            'reviewer_id': self.reviewer_id,
            'timestamp': self.timestamp,
            'corrected_label': self.corrected_label,
            'corrected_class_id': self.corrected_class_id,
            'confidence_in_review': self.confidence_in_review,
            'notes': self.notes,
            'flagged_for_escalation': self.flagged_for_escalation,
            'additional_annotations': self.additional_annotations
        }


@dataclass
class ReviewSession:
    """Container for an expert review session"""
    session_id: str
    reviewer_id: str
    start_time: float
    end_time: Optional[float] = None
    items_reviewed: List[str] = field(default_factory=list)
    feedback_list: List[ExpertFeedback] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def review_rate_per_hour(self) -> float:
        if self.duration_seconds > 0:
            return len(self.items_reviewed) / (self.duration_seconds / 3600)
        return 0


class Reviewer:
    """
    Manages expert reviewer sessions and feedback collection
    
    Features:
    - Session management
    - Feedback validation
    - Quality control (confidence scoring)
    - Escalation handling
    - Reviewer performance metrics
    """
    
    def __init__(self, reviewer_id: str, expertise_level: str = 'intermediate'):
        """
        Initialize reviewer
        
        Args:
            reviewer_id: Unique identifier for the reviewer
            expertise_level: 'expert', 'intermediate', or 'trainee'
        """
        self.reviewer_id = reviewer_id
        self.expertise_level = expertise_level
        self.active_session: Optional[ReviewSession] = None
        self.review_history: List[ReviewSession] = []
        
        # Quality metrics
        self.quality_metrics = {
            'total_reviews': 0,
            'avg_confidence': 0,
            'escalation_rate': 0,
            'agreement_with_peers': 0,
            'avg_review_time_seconds': 0
        }
        
        logger.info(f"Reviewer {reviewer_id} initialized (level={expertise_level})")
        
    def start_session(self) -> str:
        """Start a new review session"""
        session_id = f"session_{self.reviewer_id}_{int(time.time())}"
        self.active_session = ReviewSession(
            session_id=session_id,
            reviewer_id=self.reviewer_id,
            start_time=time.time()
        )
        logger.info(f"Started session {session_id}")
        return session_id
    
    def end_session(self) -> Optional[ReviewSession]:
        """End current review session"""
        if self.active_session:
            self.active_session.end_time = time.time()
            self.review_history.append(self.active_session)
            session = self.active_session
            self.active_session = None
            
            # Update metrics
            self._update_metrics(session)
            logger.info(f"Ended session {session.session_id}, "
                       f"reviewed {len(session.items_reviewed)} items")
            return session
        return None
    
    def submit_feedback(self, item_id: str, 
                       corrected_label: str,
                       confidence: float = 1.0,
                       notes: str = "",
                       escalate: bool = False) -> Optional[ExpertFeedback]:
        """
        Submit feedback for a reviewed item
        
        Args:
            item_id: ID of the reviewed item
            corrected_label: Corrected class label
            confidence: Expert's confidence in their review (0-1)
            notes: Additional notes
            escalate: Flag for escalation (e.g., need second opinion)
            
        Returns:
            ExpertFeedback object
        """
        if not self.active_session:
            logger.warning("No active session. Call start_session() first.")
            return None
            
        # Validate feedback
        if not self._validate_feedback(corrected_label, confidence):
            logger.warning(f"Invalid feedback for item {item_id}")
            return None
            
        # Map label to class ID
        class_mapping = {
            'Normal': 0, 'AFib': 1, 'PVC': 2, 'Bradycardia': 3, 'Other': 4
        }
        class_id = class_mapping.get(corrected_label, 4)
        
        feedback = ExpertFeedback(
            item_id=item_id,
            reviewer_id=self.reviewer_id,
            timestamp=time.time(),
            corrected_label=corrected_label,
            corrected_class_id=class_id,
            confidence_in_review=confidence,
            notes=notes,
            flagged_for_escalation=escalate
        )
        
        # Record in session
        self.active_session.items_reviewed.append(item_id)
        self.active_session.feedback_list.append(feedback)
        
        logger.info(f"Feedback submitted for item {item_id}: {corrected_label} "
                   f"(confidence={confidence}, escalate={escalate})")
        
        return feedback
    
    def _validate_feedback(self, corrected_label: str, confidence: float) -> bool:
        """Validate expert feedback"""
        valid_labels = ['Normal', 'AFib', 'PVC', 'Bradycardia', 'Other']
        
        if corrected_label not in valid_labels:
            logger.warning(f"Invalid label: {corrected_label}")
            return False
            
        if not 0 <= confidence <= 1:
            logger.warning(f"Invalid confidence: {confidence}")
            return False
            
        return True
    
    def _update_metrics(self, session: ReviewSession):
        """Update reviewer performance metrics"""
        n_reviews = len(session.feedback_list)
        
        if n_reviews == 0:
            return
            
        # Update total reviews
        self.quality_metrics['total_reviews'] += n_reviews
        
        # Update average confidence
        avg_conf = np.mean([f.confidence_in_review for f in session.feedback_list])
        alpha = 0.1  # EMA smoothing
        self.quality_metrics['avg_confidence'] = (
            alpha * avg_conf + 
            (1 - alpha) * self.quality_metrics['avg_confidence']
        )
        
        # Update escalation rate
        escalation_count = sum(1 for f in session.feedback_list if f.flagged_for_escalation)
        escalation_rate = escalation_count / n_reviews
        self.quality_metrics['escalation_rate'] = (
            alpha * escalation_rate +
            (1 - alpha) * self.quality_metrics['escalation_rate']
        )
        
        # Update average review time
        if session.duration_seconds > 0:
            avg_time = session.duration_seconds / n_reviews
            self.quality_metrics['avg_review_time_seconds'] = (
                alpha * avg_time +
                (1 - alpha) * self.quality_metrics['avg_review_time_seconds']
            )
            
    def get_performance_report(self) -> Dict:
        """Get reviewer performance report"""
        report = {
            'reviewer_id': self.reviewer_id,
            'expertise_level': self.expertise_level,
            'metrics': self.quality_metrics,
            'session_history': [
                {
                    'session_id': s.session_id,
                    'items_reviewed': len(s.items_reviewed),
                    'duration_seconds': s.duration_seconds,
                    'review_rate_per_hour': s.review_rate_per_hour
                }
                for s in self.review_history[-10:]  # Last 10 sessions
            ]
        }
        return report


class BatchReviewer:
    """
    Batch reviewer for processing multiple items at once
    Useful for offline review or periodic review sessions
    """
    
    def __init__(self, review_queue, feedback_store, batch_size: int = 10):
        """
        Initialize batch reviewer
        
        Args:
            review_queue: ReviewQueue instance
            feedback_store: FeedbackStore instance
            batch_size: Number of items to process in one batch
        """
        self.review_queue = review_queue
        self.feedback_store = feedback_store
        self.batch_size = batch_size
        
    def get_batch(self) -> List[ReviewItem]:
        """Get a batch of items for review"""
        items = []
        
        for _ in range(self.batch_size):
            item = self.review_queue.get_next_item()
            if item:
                items.append(item)
            else:
                break
                
        return items
    
    def submit_batch(self, reviewer_id: str,
                    reviews: List[Dict]) -> List[ExpertFeedback]:
        """
        Submit a batch of reviews
        
        Args:
            reviewer_id: ID of the reviewer
            reviews: List of review dictionaries with item_id and feedback
            
        Returns:
            List of ExpertFeedback objects
        """
        reviewer = Reviewer(reviewer_id)
        reviewer.start_session()
        
        feedback_list = []
        
        for review in reviews:
            feedback = reviewer.submit_feedback(
                item_id=review['item_id'],
                corrected_label=review['corrected_label'],
                confidence=review.get('confidence', 1.0),
                notes=review.get('notes', ''),
                escalate=review.get('escalate', False)
            )
            
            if feedback:
                feedback_list.append(feedback)
                
                # Mark queue item as completed
                self.review_queue.complete_review(
                    review['item_id'],
                    {'corrected_label': review['corrected_label'], **review}
                )
                
                # Store feedback
                self.feedback_store.add_feedback(feedback)
                
        reviewer.end_session()
        
        return feedback_list


if __name__ == "__main__":
    # Test reviewer
    reviewer = Reviewer(reviewer_id="dr_smith", expertise_level="expert")
    
    # Start session
    session_id = reviewer.start_session()
    print(f"Started session: {session_id}")
    
    # Submit feedback
    feedback = reviewer.submit_feedback(
        item_id="test_item_001",
        corrected_label="Normal",
        confidence=0.95,
        notes="Clear normal sinus rhythm"
    )
    
    if feedback:
        print(f"Submitted feedback for {feedback.item_id}")
        
    # End session
    session = reviewer.end_session()
    print(f"Session ended. Reviewed {len(session.items_reviewed)} items")
    
    # Get performance report
    report = reviewer.get_performance_report()
    print(f"Performance: {json.dumps(report['metrics'], indent=2)}")