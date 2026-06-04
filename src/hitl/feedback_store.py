"""
Storage and management of expert feedback for continuous learning
"""

import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Container for stored feedback record"""
    id: str
    item_id: str
    reviewer_id: str
    timestamp: float
    original_prediction: Dict[str, Any]
    corrected_label: str
    confidence_in_review: float
    notes: str = ""
    flagged_for_escalation: bool = False
    used_for_training: bool = False
    training_epoch: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'item_id': self.item_id,
            'reviewer_id': self.reviewer_id,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'original_prediction': self.original_prediction,
            'corrected_label': self.corrected_label,
            'confidence_in_review': self.confidence_in_review,
            'notes': self.notes,
            'flagged_for_escalation': self.flagged_for_escalation,
            'used_for_training': self.used_for_training,
            'training_epoch': self.training_epoch
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FeedbackRecord':
        return cls(**data)


@dataclass
class FeedbackStats:
    """Statistics for feedback data"""
    total_feedback: int = 0
    used_for_training: int = 0
    by_reviewer: Dict[str, int] = field(default_factory=dict)
    by_class: Dict[str, int] = field(default_factory=dict)
    by_original_class: Dict[str, int] = field(default_factory=dict)
    correction_rate: float = 0.0
    escalation_rate: float = 0.0
    avg_reviewer_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'total_feedback': self.total_feedback,
            'used_for_training': self.used_for_training,
            'by_reviewer': self.by_reviewer,
            'by_class': self.by_class,
            'by_original_class': self.by_original_class,
            'correction_rate': self.correction_rate,
            'escalation_rate': self.escalation_rate,
            'avg_reviewer_confidence': self.avg_reviewer_confidence
        }


class FeedbackStore:
    """
    Persistent storage for expert feedback
    
    Features:
    - JSON/File-based persistence
    - Query and filtering
    - Statistical analysis
    - Export for model retraining
    - Feedback quality metrics
    """
    
    def __init__(self, storage_path: str = "./data/feedback"):
        """
        Initialize feedback store
        
        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.feedback_records: List[FeedbackRecord] = []
        self._load_records()
        
        logger.info(f"FeedbackStore initialized at {storage_path}")
        
    def add_feedback(self, feedback: ExpertFeedback, 
                    original_prediction: Dict[str, Any]) -> str:
        """
        Add new feedback record
        
        Args:
            feedback: ExpertFeedback object
            original_prediction: Original model prediction
            
        Returns:
            Record ID
        """
        import uuid
        
        record_id = str(uuid.uuid4())
        
        record = FeedbackRecord(
            id=record_id,
            item_id=feedback.item_id,
            reviewer_id=feedback.reviewer_id,
            timestamp=feedback.timestamp,
            original_prediction=original_prediction,
            corrected_label=feedback.corrected_label,
            confidence_in_review=feedback.confidence_in_review,
            notes=feedback.notes,
            flagged_for_escalation=feedback.flagged_for_escalation
        )
        
        self.feedback_records.append(record)
        self._save_records()
        
        logger.info(f"Added feedback record {record_id} for item {feedback.item_id}")
        return record_id
    
    def get_feedback(self, record_id: str) -> Optional[FeedbackRecord]:
        """Get feedback record by ID"""
        for record in self.feedback_records:
            if record.id == record_id:
                return record
        return None
    
    def get_feedback_by_item(self, item_id: str) -> List[FeedbackRecord]:
        """Get all feedback for a specific item"""
        return [r for r in self.feedback_records if r.item_id == item_id]
    
    def get_feedback_by_reviewer(self, reviewer_id: str) -> List[FeedbackRecord]:
        """Get all feedback from a specific reviewer"""
        return [r for r in self.feedback_records if r.reviewer_id == reviewer_id]
    
    def get_feedback_by_date_range(self, start_time: float, 
                                   end_time: float) -> List[FeedbackRecord]:
        """Get feedback within date range"""
        return [r for r in self.feedback_records 
                if start_time <= r.timestamp <= end_time]
    
    def get_feedback_for_training(self, min_confidence: float = 0.7,
                                  limit: Optional[int] = None) -> List[FeedbackRecord]:
        """
        Get feedback records suitable for model training
        
        Args:
            min_confidence: Minimum reviewer confidence threshold
            limit: Maximum number of records to return
            
        Returns:
            List of FeedbackRecord objects
        """
        eligible = [
            r for r in self.feedback_records 
            if not r.used_for_training and r.confidence_in_review >= min_confidence
        ]
        
        # Sort by confidence (highest first)
        eligible.sort(key=lambda x: x.confidence_in_review, reverse=True)
        
        if limit:
            eligible = eligible[:limit]
            
        return eligible
    
    def mark_as_trained(self, record_ids: List[str], epoch: int):
        """Mark feedback records as used for training"""
        for record_id in record_ids:
            record = self.get_feedback(record_id)
            if record:
                record.used_for_training = True
                record.training_epoch = epoch
                
        self._save_records()
        logger.info(f"Marked {len(record_ids)} records as trained at epoch {epoch}")
    
    def get_stats(self) -> FeedbackStats:
        """Compute feedback statistics"""
        stats = FeedbackStats()
        
        if not self.feedback_records:
            return stats
            
        stats.total_feedback = len(self.feedback_records)
        
        # Count by reviewer
        for record in self.feedback_records:
            stats.by_reviewer[record.reviewer_id] = \
                stats.by_reviewer.get(record.reviewer_id, 0) + 1
                
        # Count by corrected class
        for record in self.feedback_records:
            stats.by_class[record.corrected_label] = \
                stats.by_class.get(record.corrected_label, 0) + 1
                
        # Count by original prediction class
        for record in self.feedback_records:
            original_class = record.original_prediction.get('class_name', 'Unknown')
            stats.by_original_class[original_class] = \
                stats.by_original_class.get(original_class, 0) + 1
                
        # Compute correction rate (how often expert disagreed with model)
        corrections = 0
        for record in self.feedback_records:
            original_class = record.original_prediction.get('class_name', 'Unknown')
            if original_class != record.corrected_label:
                corrections += 1
        stats.correction_rate = corrections / stats.total_feedback
        
        # Compute escalation rate
        escalations = sum(1 for r in self.feedback_records if r.flagged_for_escalation)
        stats.escalation_rate = escalations / stats.total_feedback
        
        # Compute average reviewer confidence
        avg_conf = np.mean([r.confidence_in_review for r in self.feedback_records])
        stats.avg_reviewer_confidence = avg_conf
        
        # Count used for training
        stats.used_for_training = sum(1 for r in self.feedback_records if r.used_for_training)
        
        return stats
    
    def get_confusion_analysis(self) -> Dict[str, Any]:
        """
        Analyze confusion patterns between model and experts
        
        Returns:
            Dictionary with confusion analysis
        """
        confusion = defaultdict(lambda: defaultdict(int))
        
        for record in self.feedback_records:
            original = record.original_prediction.get('class_name', 'Unknown')
            corrected = record.corrected_label
            confusion[original][corrected] += 1
            
        # Calculate misclassification rates
        misclassification_rate = {}
        for original, corrections in confusion.items():
            total = sum(corrections.values())
            if total > 0:
                misclassifications = sum(corrections[c] for c in corrections if c != original)
                misclassification_rate[original] = misclassifications / total
                
        return {
            'confusion_matrix': dict(confusion),
            'misclassification_rate': misclassification_rate
        }
    
    def export_for_training(self, output_path: str, 
                           include_features: bool = False,
                           features_store=None):
        """
        Export feedback data for model retraining
        
        Args:
            output_path: Path to save training data
            include_features: Include feature vectors
            features_store: Store containing feature vectors
        """
        export_data = []
        
        for record in self.feedback_records:
            if not record.used_for_training:
                item = {
                    'feedback_id': record.id,
                    'corrected_label': record.corrected_label,
                    'reviewer_confidence': record.confidence_in_review,
                    'original_prediction': record.original_prediction
                }
                
                if include_features and features_store:
                    features = features_store.get_features(record.item_id)
                    if features is not None:
                        item['features'] = features.tolist()
                        
                export_data.append(item)
                
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported {len(export_data)} records to {output_path}")
        
    def _save_records(self):
        """Save records to disk"""
        filepath = self.storage_path / "feedback_records.json"
        
        data = {
            'records': [r.to_dict() for r in self.feedback_records],
            'last_updated': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def _load_records(self):
        """Load records from disk"""
        filepath = self.storage_path / "feedback_records.json"
        
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                self.feedback_records = [
                    FeedbackRecord.from_dict(r) for r in data.get('records', [])
                ]
                
                logger.info(f"Loaded {len(self.feedback_records)} feedback records")
                
            except Exception as e:
                logger.error(f"Failed to load records: {e}")
                self.feedback_records = []
                
    def clear_storage(self):
        """Clear all stored feedback (use with caution)"""
        self.feedback_records = []
        self._save_records()
        logger.warning("Cleared all feedback records")


class FeedbackAnalytics:
    """
    Analytics and reporting for feedback data
    """
    
    def __init__(self, feedback_store: FeedbackStore):
        """
        Initialize feedback analytics
        
        Args:
            feedback_store: FeedbackStore instance
        """
        self.store = feedback_store
        
    def generate_report(self) -> Dict:
        """
        Generate comprehensive feedback report
        
        Returns:
            Dictionary with report data
        """
        stats = self.store.get_stats()
        confusion = self.store.get_confusion_analysis()
        
        report = {
            'summary': stats.to_dict(),
            'confusion_analysis': confusion,
            'reviewer_performance': self._get_reviewer_performance(),
            'temporal_trends': self._get_temporal_trends(),
            'recommendations': self._generate_recommendations(stats, confusion)
        }
        
        return report
    
    def _get_reviewer_performance(self) -> Dict:
        """Get performance metrics per reviewer"""
        performance = {}
        
        for reviewer_id in set(r.reviewer_id for r in self.store.feedback_records):
            reviewer_records = self.store.get_feedback_by_reviewer(reviewer_id)
            
            performance[reviewer_id] = {
                'total_reviews': len(reviewer_records),
                'avg_confidence': np.mean([r.confidence_in_review for r in reviewer_records]),
                'escalation_rate': sum(1 for r in reviewer_records if r.flagged_for_escalation) / len(reviewer_records),
                'correction_rate': sum(1 for r in reviewer_records 
                                      if r.corrected_label != r.original_prediction.get('class_name')) / len(reviewer_records)
            }
            
        return performance
    
    def _get_temporal_trends(self) -> Dict:
        """Get temporal trends in feedback"""
        if not self.store.feedback_records:
            return {}
            
        # Group by week
        records_by_week = defaultdict(list)
        
        for record in self.store.feedback_records:
            week = int(record.timestamp / (7 * 86400))
            records_by_week[week].append(record)
            
        trends = {
            'correction_rate_over_time': [],
            'reviewer_confidence_over_time': [],
            'feedback_volume_over_time': []
        }
        
        for week in sorted(records_by_week.keys()):
            week_records = records_by_week[week]
            
            correction_rate = sum(1 for r in week_records 
                                 if r.corrected_label != r.original_prediction.get('class_name')) / len(week_records)
            avg_confidence = np.mean([r.confidence_in_review for r in week_records])
            
            trends['correction_rate_over_time'].append({
                'week': week,
                'value': correction_rate
            })
            trends['reviewer_confidence_over_time'].append({
                'week': week,
                'value': avg_confidence
            })
            trends['feedback_volume_over_time'].append({
                'week': week,
                'value': len(week_records)
            })
            
        return trends
    
    def _generate_recommendations(self, stats: FeedbackStats, 
                                  confusion: Dict) -> List[str]:
        """Generate recommendations based on feedback analysis"""
        recommendations = []
        
        # Check if correction rate is high
        if stats.correction_rate > 0.3:
            recommendations.append(
                f"High correction rate ({stats.correction_rate:.1%}). "
                "Consider model retraining or threshold adjustment."
            )
            
        # Check for problematic classes
        for original, rate in confusion.get('misclassification_rate', {}).items():
            if rate > 0.4:
                recommendations.append(
                    f"Class '{original}' has high misclassification rate ({rate:.1%}). "
                    "Collect more labeled samples for this class."
                )
                
        # Check if escalation rate is high
        if stats.escalation_rate > 0.2:
            recommendations.append(
                f"High escalation rate ({stats.escalation_rate:.1%}). "
                "Consider adding more expert reviewers or improving model confidence."
            )
            
        # Check if enough feedback for retraining
        if stats.total_feedback - stats.used_for_training > 500:
            recommendations.append(
                f"{stats.total_feedback - stats.used_for_training} new feedback samples available. "
                "Consider retraining the model."
            )
            
        return recommendations


if __name__ == "__main__":
    # Test feedback store
    store = FeedbackStore()
    
    # Add sample feedback
    for i in range(10):
        import uuid
        feedback = ExpertFeedback(
            item_id=f"item_{i}",
            reviewer_id="dr_smith",
            timestamp=time.time(),
            corrected_label=np.random.choice(['Normal', 'AFib', 'PVC']),
            confidence_in_review=0.9,
            notes="Sample feedback"
        )
        
        original_prediction = {
            'class_name': np.random.choice(['Normal', 'AFib', 'PVC', 'Other']),
            'confidence': np.random.random()
        }
        
        store.add_feedback(feedback, original_prediction)
        
    # Get stats
    stats = store.get_stats()
    print(f"Feedback Stats:")
    print(f"  Total: {stats.total_feedback}")
    print(f"  Correction rate: {stats.correction_rate:.2%}")
    
    # Generate analytics report
    analytics = FeedbackAnalytics(store)
    report = analytics.generate_report()
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")