"""
Review queue management for human expert review of low-confidence predictions
"""

import json
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewPriority(Enum):
    """Priority levels for review items"""
    CRITICAL = 0   # Immediate attention (very low confidence, anomaly)
    HIGH = 1       # Review soon
    MEDIUM = 2     # Normal priority
    LOW = 3        # Low priority, can batch


class ReviewStatus(Enum):
    """Status of review items"""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    SKIPPED = "skipped"


@dataclass
class ReviewItem:
    """Container for items awaiting expert review"""
    id: str
    timestamp: float
    beat_data: np.ndarray  # ECG beat waveform
    model_prediction: Dict[str, Any]  # Original model output
    priority: ReviewPriority
    status: ReviewStatus
    assigned_to: Optional[str] = None
    review_start_time: Optional[float] = None
    review_complete_time: Optional[float] = None
    expert_feedback: Optional[Dict] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to serializable dictionary"""
        data = asdict(self)
        # Convert numpy array to list for JSON serialization
        if isinstance(data['beat_data'], np.ndarray):
            data['beat_data'] = data['beat_data'].tolist()
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ReviewItem':
        """Create from dictionary"""
        data = data.copy()
        data['beat_data'] = np.array(data['beat_data'])
        data['priority'] = ReviewPriority(data['priority'])
        data['status'] = ReviewStatus(data['status'])
        return cls(**data)


@dataclass
class QueueConfig:
    """Configuration for review queue"""
    max_size: int = 1000
    critical_timeout_seconds: int = 300  # 5 minutes for critical
    high_timeout_seconds: int = 3600     # 1 hour for high
    medium_timeout_seconds: int = 86400  # 24 hours for medium
    low_timeout_seconds: int = 604800    # 7 days for low
    enable_escalation: bool = True
    auto_skip_after_timeout: bool = False
    persistence_enabled: bool = True
    persistence_file: str = "./data/review_queue.json"


class ReviewQueue:
    """
    Manages the queue of items waiting for expert review
    
    Features:
    - Priority-based queue management
    - Timeout and escalation handling
    - Persistent storage
    - Batch processing support
    - Redis backend support (optional)
    """
    
    def __init__(self, config: QueueConfig, use_redis: bool = False, 
                 redis_url: Optional[str] = None):
        """
        Initialize review queue
        
        Args:
            config: Queue configuration
            use_redis: Use Redis backend (for distributed systems)
            redis_url: Redis connection URL if use_redis is True
        """
        self.config = config
        self.use_redis = use_redis
        
        # Initialize storage
        if use_redis:
            self._init_redis(redis_url)
        else:
            # In-memory queues by priority
            self.queues = {
                ReviewPriority.CRITICAL: deque(),
                ReviewPriority.HIGH: deque(),
                ReviewPriority.MEDIUM: deque(),
                ReviewPriority.LOW: deque()
            }
            self.items_by_id = {}
            
        # Load persisted items
        if config.persistence_enabled and not use_redis:
            self._load_persistence()
            
        logger.info(f"ReviewQueue initialized (redis={use_redis}, max_size={config.max_size})")
        
    def _init_redis(self, redis_url: Optional[str]):
        """Initialize Redis connection"""
        try:
            import redis
            self.redis_client = redis.from_url(redis_url or 'redis://localhost:6379/0')
            logger.info("Redis connection established")
        except ImportError:
            logger.error("Redis package not installed. Falling back to in-memory.")
            self.use_redis = False
            self.queues = {p: deque() for p in ReviewPriority}
            self.items_by_id = {}
            
    def add_item(self, beat_data: np.ndarray, 
                 model_prediction: Dict[str, Any],
                 priority: Optional[ReviewPriority] = None,
                 metadata: Optional[Dict] = None) -> str:
        """
        Add a new item to the review queue
        
        Args:
            beat_data: ECG beat waveform
            model_prediction: Original model prediction output
            priority: Review priority (auto-determined if None)
            metadata: Additional metadata
            
        Returns:
            Review item ID
        """
        # Auto-determine priority if not specified
        if priority is None:
            priority = self._determine_priority(model_prediction)
            
        # Check queue size limit
        total_size = self._get_total_size()
        if total_size >= self.config.max_size:
            logger.warning(f"Queue full ({total_size}/{self.config.max_size}), dropping lowest priority")
            self._drop_lowest_priority()
            
        # Create review item
        review_id = str(uuid.uuid4())
        item = ReviewItem(
            id=review_id,
            timestamp=time.time(),
            beat_data=beat_data,
            model_prediction=model_prediction,
            priority=priority,
            status=ReviewStatus.PENDING,
            metadata=metadata
        )
        
        # Add to queue
        if self.use_redis:
            self._redis_add_item(item)
        else:
            self.queues[priority].append(item)
            self.items_by_id[review_id] = item
            
        logger.info(f"Added review item {review_id} with priority {priority.name}")
        
        # Persist if needed
        if self.config.persistence_enabled and not self.use_redis:
            self._save_persistence()
            
        return review_id
    
    def _determine_priority(self, model_prediction: Dict[str, Any]) -> ReviewPriority:
        """Automatically determine review priority based on model output"""
        confidence = model_prediction.get('confidence', 0.5)
        anomaly_score = model_prediction.get('anomaly_score', 0)
        predicted_class = model_prediction.get('class_name', 'Unknown')
        
        # Critical: Very low confidence or high anomaly score
        if confidence < 0.3 or anomaly_score > 0.8:
            return ReviewPriority.CRITICAL
            
        # High: Low confidence or moderate anomaly
        if confidence < 0.5 or anomaly_score > 0.6:
            return ReviewPriority.HIGH
            
        # Medium: Borderline cases
        if confidence < 0.7 or anomaly_score > 0.4:
            return ReviewPriority.MEDIUM
            
        # Check for critical arrhythmias
        if predicted_class in ['AFib', 'PVC'] and confidence < 0.8:
            return ReviewPriority.HIGH
            
        return ReviewPriority.LOW
    
    def get_next_item(self, reviewer_id: Optional[str] = None) -> Optional[ReviewItem]:
        """
        Get the next item for review (highest priority first)
        
        Args:
            reviewer_id: Identifier for the reviewer (for assignment tracking)
            
        Returns:
            ReviewItem or None if queue is empty
        """
        # Check priorities in order
        for priority in [ReviewPriority.CRITICAL, ReviewPriority.HIGH, 
                        ReviewPriority.MEDIUM, ReviewPriority.LOW]:
            item = self._pop_from_queue(priority)
            if item:
                # Update status
                item.status = ReviewStatus.IN_REVIEW
                item.review_start_time = time.time()
                item.assigned_to = reviewer_id
                
                # Update in storage
                if not self.use_redis:
                    self.items_by_id[item.id] = item
                    
                logger.info(f"Item {item.id} assigned to {reviewer_id}")
                return item
                
        return None
    
    def _pop_from_queue(self, priority: ReviewPriority) -> Optional[ReviewItem]:
        """Pop an item from a specific priority queue"""
        if self.use_redis:
            return self._redis_pop_item(priority)
        else:
            if self.queues[priority]:
                return self.queues[priority].popleft()
        return None
    
    def complete_review(self, item_id: str, 
                       expert_feedback: Dict[str, Any]) -> bool:
        """
        Mark a review item as completed with expert feedback
        
        Args:
            item_id: ID of the review item
            expert_feedback: Feedback from expert (corrected label, notes, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        item = self.get_item(item_id)
        if not item:
            logger.warning(f"Item {item_id} not found")
            return False
            
        item.status = ReviewStatus.COMPLETED
        item.review_complete_time = time.time()
        item.expert_feedback = expert_feedback
        
        # Save
        if not self.use_redis:
            self.items_by_id[item_id] = item
            
        if self.config.persistence_enabled and not self.use_redis:
            self._save_persistence()
            
        logger.info(f"Item {item_id} completed review")
        return True
    
    def get_item(self, item_id: str) -> Optional[ReviewItem]:
        """Get a specific review item by ID"""
        if self.use_redis:
            return self._redis_get_item(item_id)
        else:
            return self.items_by_id.get(item_id)
    
    def get_pending_items(self, limit: Optional[int] = None) -> List[ReviewItem]:
        """Get all pending items (optionally limited)"""
        items = []
        
        if self.use_redis:
            items = self._redis_get_all_pending()
        else:
            for priority in ReviewPriority:
                items.extend(list(self.queues[priority]))
                
        if limit:
            items = items[:limit]
            
        return items
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        stats = {
            'total_pending': self._get_total_size(),
            'by_priority': {},
            'by_status': {},
            'oldest_item_age': 0
        }
        
        # Count by priority
        for priority in ReviewPriority:
            if self.use_redis:
                count = self._redis_get_priority_count(priority)
            else:
                count = len(self.queues[priority])
            stats['by_priority'][priority.name] = count
            
        # Count by status
        if not self.use_redis:
            for item in self.items_by_id.values():
                stats['by_status'][item.status.value] = \
                    stats['by_status'].get(item.status.value, 0) + 1
                    
            # Oldest item
            oldest_time = None
            for item in self.items_by_id.values():
                if item.status == ReviewStatus.PENDING:
                    if oldest_time is None or item.timestamp < oldest_time:
                        oldest_time = item.timestamp
                        
            if oldest_time:
                stats['oldest_item_age'] = time.time() - oldest_time
                
        return stats
    
    def _get_total_size(self) -> int:
        """Get total number of pending items"""
        if self.use_redis:
            return self._redis_get_total_size()
        else:
            return sum(len(q) for q in self.queues.values())
    
    def _drop_lowest_priority(self):
        """Drop the oldest lowest priority item when queue is full"""
        for priority in [ReviewPriority.LOW, ReviewPriority.MEDIUM, 
                        ReviewPriority.HIGH, ReviewPriority.CRITICAL]:
            if self.use_redis:
                item = self._redis_pop_item(priority)
            else:
                if self.queues[priority]:
                    item = self.queues[priority].popleft()
                    del self.items_by_id[item.id]
                    
            if item:
                logger.info(f"Dropped item {item.id} from {priority.name} queue")
                break
                
    def check_timeouts(self):
        """Check for timed-out items and escalate if needed"""
        if not self.config.enable_escalation:
            return
            
        current_time = time.time()
        timeout_map = {
            ReviewPriority.CRITICAL: self.config.critical_timeout_seconds,
            ReviewPriority.HIGH: self.config.high_timeout_seconds,
            ReviewPriority.MEDIUM: self.config.medium_timeout_seconds,
            ReviewPriority.LOW: self.config.low_timeout_seconds
        }
        
        for item in self.get_pending_items():
            age = current_time - item.timestamp
            timeout = timeout_map.get(item.priority, 3600)
            
            if age > timeout:
                if self.config.auto_skip_after_timeout:
                    item.status = ReviewStatus.SKIPPED
                    logger.info(f"Item {item.id} auto-skipped after timeout")
                else:
                    # Escalate to next priority
                    old_priority = item.priority
                    if item.priority == ReviewPriority.LOW:
                        item.priority = ReviewPriority.MEDIUM
                    elif item.priority == ReviewPriority.MEDIUM:
                        item.priority = ReviewPriority.HIGH
                    elif item.priority == ReviewPriority.HIGH:
                        item.priority = ReviewPriority.CRITICAL
                        
                    logger.info(f"Item {item.id} escalated from {old_priority.name} to {item.priority.name}")
                    
    def _save_persistence(self):
        """Save queue state to disk"""
        try:
            data = {
                'items': [item.to_dict() for item in self.items_by_id.values() 
                         if item.status == ReviewStatus.PENDING],
                'timestamp': time.time()
            }
            with open(self.config.persistence_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save persistence: {e}")
            
    def _load_persistence(self):
        """Load queue state from disk"""
        try:
            import os
            if os.path.exists(self.config.persistence_file):
                with open(self.config.persistence_file, 'r') as f:
                    data = json.load(f)
                    
                for item_data in data.get('items', []):
                    item = ReviewItem.from_dict(item_data)
                    if item.status == ReviewStatus.PENDING:
                        self.queues[item.priority].append(item)
                        self.items_by_id[item.id] = item
                        
                logger.info(f"Loaded {len(data.get('items', []))} items from persistence")
        except Exception as e:
            logger.error(f"Failed to load persistence: {e}")
    
    def _redis_add_item(self, item: ReviewItem):
        """Add item to Redis"""
        key = f"review_queue:{item.priority.value}"
        self.redis_client.lpush(key, json.dumps(item.to_dict()))
        self.redis_client.hset("review_items", item.id, json.dumps(item.to_dict()))
        
    def _redis_pop_item(self, priority: ReviewPriority) -> Optional[ReviewItem]:
        """Pop item from Redis"""
        key = f"review_queue:{priority.value}"
        data = self.redis_client.rpop(key)
        if data:
            item_data = json.loads(data)
            return ReviewItem.from_dict(item_data)
        return None
    
    def _redis_get_item(self, item_id: str) -> Optional[ReviewItem]:
        """Get item from Redis by ID"""
        data = self.redis_client.hget("review_items", item_id)
        if data:
            return ReviewItem.from_dict(json.loads(data))
        return None
    
    def _redis_get_priority_count(self, priority: ReviewPriority) -> int:
        """Get count for a priority in Redis"""
        key = f"review_queue:{priority.value}"
        return self.redis_client.llen(key)
    
    def _redis_get_total_size(self) -> int:
        """Get total size from Redis"""
        total = 0
        for priority in ReviewPriority:
            total += self._redis_get_priority_count(priority)
        return total
    
    def _redis_get_all_pending(self) -> List[ReviewItem]:
        """Get all pending items from Redis"""
        items = []
        for priority in ReviewPriority:
            key = f"review_queue:{priority.value}"
            for i in range(self._redis_get_priority_count(priority)):
                data = self.redis_client.lindex(key, i)
                if data:
                    items.append(ReviewItem.from_dict(json.loads(data)))
        return items


if __name__ == "__main__":
    # Test review queue
    config = QueueConfig(max_size=100)
    queue = ReviewQueue(config)
    
    # Add sample items
    for i in range(5):
        item_id = queue.add_item(
            beat_data=np.random.randn(187),
            model_prediction={'confidence': 0.5 + i*0.1, 'class_name': 'Normal'},
            metadata={'source': 'test', 'index': i}
        )
        print(f"Added item: {item_id}")
        
    # Get stats
    stats = queue.get_stats()
    print(f"Queue stats: {stats}")
    
    # Get next item
    item = queue.get_next_item(reviewer_id="test_reviewer")
    if item:
        print(f"Got item: {item.id}, priority: {item.priority.name}")
        
        # Complete review
        queue.complete_review(item.id, {'corrected_label': 'Normal', 'notes': 'Looks normal'})