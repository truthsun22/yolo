import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from app.config import settings
from app.models import Event, EventType, EventCreate


@dataclass
class TrackedObject:
    track_id: int
    first_seen: datetime
    last_seen: datetime
    last_event_time: Optional[datetime] = None
    event_count: int = 0
    bounding_boxes: List[Dict[str, float]] = field(default_factory=list)


class EventManager:
    _global_event_cache: Dict[str, Dict[int, TrackedObject]] = {}
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self._tracked_objects: Dict[int, TrackedObject] = self._get_task_cache()
        self._events_file = os.path.join(settings.EVENTS_DIR, f"{task_id}_events.json")
        self._events: List[Event] = []
        self._load_events_from_disk()
    
    def _get_task_cache(self) -> Dict[int, TrackedObject]:
        if self.task_id not in EventManager._global_event_cache:
            EventManager._global_event_cache[self.task_id] = {}
        return EventManager._global_event_cache[self.task_id]
    
    def _load_events_from_disk(self):
        if os.path.exists(self._events_file):
            try:
                with open(self._events_file, 'r', encoding='utf-8') as f:
                    events_data = json.load(f)
                for event_data in events_data:
                    if 'timestamp' in event_data and isinstance(event_data['timestamp'], str):
                        event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                    event = Event(**event_data)
                    self._events.append(event)
            except Exception as e:
                from app.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"加载事件数据失败: {str(e)}")
    
    def _save_event_to_disk(self, event: Event):
        try:
            self._events.append(event)
            events_data = []
            for e in self._events:
                event_dict = e.model_dump()
                event_dict['timestamp'] = event_dict['timestamp'].isoformat()
                events_data.append(event_dict)
            with open(self._events_file, 'w', encoding='utf-8') as f:
                json.dump(events_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            from app.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"保存事件到磁盘失败: {str(e)}")
    
    def should_generate_event(
        self,
        track_id: Optional[int],
        confidence: float,
        bounding_box: Dict[str, float],
        frame_number: int
    ) -> tuple[bool, Optional[str]]:
        now = datetime.now()
        
        if track_id is None:
            return self._should_generate_event_for_untracked(
                confidence, bounding_box, frame_number, now
            )
        
        return self._should_generate_event_for_tracked(
            track_id, confidence, bounding_box, frame_number, now
        )
    
    def _should_generate_event_for_untracked(
        self,
        confidence: float,
        bounding_box: Dict[str, float],
        frame_number: int,
        now: datetime
    ) -> tuple[bool, Optional[str]]:
        if confidence < settings.MODEL_CONFIDENCE_THRESHOLD:
            return False, None
        
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        return True, event_id
    
    def _should_generate_event_for_tracked(
        self,
        track_id: int,
        confidence: float,
        bounding_box: Dict[str, float],
        frame_number: int,
        now: datetime
    ) -> tuple[bool, Optional[str]]:
        if confidence < settings.MODEL_CONFIDENCE_THRESHOLD:
            return False, None
        
        tracked = self._tracked_objects.get(track_id)
        
        if tracked is None:
            tracked = TrackedObject(
                track_id=track_id,
                first_seen=now,
                last_seen=now
            )
            self._tracked_objects[track_id] = tracked
            
            event_id = f"evt_{uuid.uuid4().hex[:12]}"
            tracked.last_event_time = now
            tracked.event_count = 1
            tracked.bounding_boxes.append(bounding_box)
            return True, event_id
        
        tracked.last_seen = now
        tracked.bounding_boxes.append(bounding_box)
        
        cooldown_period = timedelta(seconds=settings.EVENT_COOLDOWN_SECONDS)
        
        if tracked.last_event_time is None:
            event_id = f"evt_{uuid.uuid4().hex[:12]}"
            tracked.last_event_time = now
            tracked.event_count += 1
            return True, event_id
        
        time_since_last_event = now - tracked.last_event_time
        
        if time_since_last_event >= cooldown_period:
            event_id = f"evt_{uuid.uuid4().hex[:12]}"
            tracked.last_event_time = now
            tracked.event_count += 1
            return True, event_id
        
        return False, None
    
    def create_event(
        self,
        event_type: EventType,
        frame_number: int,
        confidence: float,
        bounding_box: Dict[str, float],
        track_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Event]:
        should_create, event_id = self.should_generate_event(
            track_id, confidence, bounding_box, frame_number
        )
        
        if not should_create or event_id is None:
            return None
        
        event = Event(
            id=event_id,
            task_id=self.task_id,
            event_type=event_type,
            timestamp=datetime.now(),
            frame_number=frame_number,
            confidence=confidence,
            bounding_box=bounding_box,
            track_id=track_id,
            metadata=metadata
        )
        
        self._save_event_to_disk(event)
        
        return event
    
    def get_events(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        events = self._events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    def get_event(self, event_id: str) -> Optional[Event]:
        for event in self._events:
            if event.id == event_id:
                return event
        return None
    
    @classmethod
    def clear_task_cache(cls, task_id: str):
        if task_id in cls._global_event_cache:
            del cls._global_event_cache[task_id]
    
    @classmethod
    def get_all_events(cls, task_ids: Optional[List[str]] = None, limit: int = 100) -> List[Event]:
        all_events = []
        
        if task_ids is None:
            if not os.path.exists(settings.EVENTS_DIR):
                return []
            for filename in os.listdir(settings.EVENTS_DIR):
                if filename.endswith('_events.json'):
                    filepath = os.path.join(settings.EVENTS_DIR, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            events_data = json.load(f)
                        for event_data in events_data:
                            if 'timestamp' in event_data and isinstance(event_data['timestamp'], str):
                                event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                            event = Event(**event_data)
                            all_events.append(event)
                    except Exception:
                        pass
        else:
            for task_id in task_ids:
                manager = EventManager(task_id)
                all_events.extend(manager.get_events())
        
        return sorted(all_events, key=lambda e: e.timestamp, reverse=True)[:limit]
