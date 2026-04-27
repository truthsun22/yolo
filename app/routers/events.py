from typing import List, Optional

from fastapi import APIRouter, Depends

from app.auth import get_current_active_user, User
from app.models import Event, EventType
from app.event_manager import EventManager
from app.logger import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix="/api/events", tags=["事件管理"])


@router.get("", response_model=List[Event])
async def list_events(
    task_id: Optional[str] = None,
    event_type: Optional[EventType] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user)
):
    if task_id:
        event_manager = EventManager(task_id)
        events = event_manager.get_events(event_type=event_type, limit=limit)
    else:
        events = EventManager.get_all_events(limit=limit)
        if event_type:
            events = [e for e in events if e.event_type == event_type]
            events = events[:limit]
    
    return events


@router.get("/{event_id}", response_model=Event)
async def get_event(
    event_id: str,
    current_user: User = Depends(get_current_active_user)
):
    events = EventManager.get_all_events(limit=1000)
    for event in events:
        if event.id == event_id:
            return event
    
    from fastapi import HTTPException, status
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"事件 {event_id} 不存在"
    )
