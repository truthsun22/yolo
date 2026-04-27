from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class TaskType(str, Enum):
    PEDESTRIAN_DETECTION = "pedestrian_detection"
    PEDESTRIAN_TRACKING = "pedestrian_tracking"


class VideoSourceType(str, Enum):
    LOCAL_FILE = "local_file"
    NETWORK_STREAM = "network_stream"


class TaskBase(BaseModel):
    name: str = Field(..., description="任务名称")
    task_type: TaskType = Field(..., description="任务类型")
    video_source: str = Field(..., description="视频源路径或URL")
    video_source_type: VideoSourceType = Field(..., description="视频源类型")
    model_name: Optional[str] = Field(None, description="使用的模型名称")
    frame_skip: int = Field(5, ge=1, le=30, description="跳帧数")
    confidence_threshold: float = Field(0.5, ge=0.1, le=1.0, description="置信度阈值")
    enable_event_generation: bool = Field(True, description="是否启用事件生成")


class TaskCreate(TaskBase):
    pass


class TaskUpdate(BaseModel):
    name: Optional[str] = None
    frame_skip: Optional[int] = None
    confidence_threshold: Optional[float] = None


class TaskResponse(TaskBase):
    id: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    progress: float = Field(0.0, ge=0.0, le=100.0)
    current_frame: int = 0
    total_frames: Optional[int] = None
    fps: Optional[float] = None
    events_count: int = 0
    detections_count: int = 0
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class EventType(str, Enum):
    PEDESTRIAN_DETECTED = "pedestrian_detected"
    PEDESTRIAN_TRACKED = "pedestrian_tracked"


class Event(BaseModel):
    id: str
    task_id: str
    event_type: EventType
    timestamp: datetime
    frame_number: int
    confidence: float
    bounding_box: Dict[str, float]
    track_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class EventCreate(BaseModel):
    task_id: str
    event_type: EventType
    frame_number: int
    confidence: float
    bounding_box: Dict[str, float]
    track_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
