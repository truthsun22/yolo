import os
from typing import Annotated, List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.auth import get_current_active_user, User
from app.config import settings
from app.models import (
    TaskCreate, TaskResponse, TaskStatus, 
    Event, EventType
)
from app.task_manager import task_manager
from app.event_manager import EventManager
from app.logger import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix="/api/tasks", tags=["任务管理"])


class VideoInfo(BaseModel):
    task_id: str
    video_path: str
    video_url: str
    exists: bool
    file_size: Optional[int] = None


class ScreenshotInfo(BaseModel):
    event_id: Optional[str] = None
    frame_number: int
    track_id: Optional[int] = None
    screenshot_path: str
    screenshot_url: str
    timestamp: str


class TaskMediaResponse(BaseModel):
    task_id: str
    video: Optional[VideoInfo] = None
    screenshots: List[ScreenshotInfo] = []


@router.post("", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    task_create: TaskCreate,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    try:
        task = task_manager.create_task(task_create)
        
        await task_manager.start_task(task.id)
        
        logger.info(f"用户 {current_user.username} 创建并启动任务: {task.id}")
        return task
    except Exception as e:
        logger.error(f"创建任务失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建任务失败: {str(e)}"
        )


@router.get("", response_model=List[TaskResponse])
async def list_tasks(
    status: Optional[TaskStatus] = None,
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    tasks = task_manager.list_tasks(status=status)
    return tasks


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务 {task_id} 不存在"
        )
    return task


@router.post("/{task_id}/start", response_model=TaskResponse)
async def start_task(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务 {task_id} 不存在"
        )
    
    if task.status == TaskStatus.RUNNING:
        return task
    
    success = await task_manager.start_task(task_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="启动任务失败"
        )
    
    logger.info(f"用户 {current_user.username} 启动任务: {task_id}")
    return task_manager.get_task(task_id)


@router.post("/{task_id}/stop", response_model=TaskResponse)
async def stop_task(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务 {task_id} 不存在"
        )
    
    if task.status not in [TaskStatus.RUNNING, TaskStatus.PAUSED]:
        return task
    
    success = await task_manager.stop_task(task_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="停止任务失败"
        )
    
    logger.info(f"用户 {current_user.username} 停止任务: {task_id}")
    return task_manager.get_task(task_id)


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务 {task_id} 不存在"
        )
    
    if task.status == TaskStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="运行中的任务无法删除，请先停止任务"
        )
    
    success = task_manager.delete_task(task_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除任务失败"
        )
    
    logger.info(f"用户 {current_user.username} 删除任务: {task_id}")


@router.get("/{task_id}/events", response_model=List[Event])
async def get_task_events(
    task_id: str,
    event_type: Optional[EventType] = None,
    limit: int = 100,
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务 {task_id} 不存在"
        )
    
    event_manager = EventManager(task_id)
    events = event_manager.get_events(event_type=event_type, limit=limit)
    return events


@router.get("/{task_id}/media", response_model=TaskMediaResponse)
async def get_task_media(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务 {task_id} 不存在"
        )
    
    video_filename = f"{task_id}_annotated.mp4"
    video_path = os.path.join(settings.VIDEO_OUTPUT_DIR, video_filename)
    video_url = f"/api/videos/{video_filename}"
    
    video_info = None
    if os.path.exists(video_path):
        file_size = os.path.getsize(video_path)
        video_info = VideoInfo(
            task_id=task_id,
            video_path=video_path,
            video_url=video_url,
            exists=True,
            file_size=file_size
        )
    else:
        video_info = VideoInfo(
            task_id=task_id,
            video_path=video_path,
            video_url=video_url,
            exists=False
        )
    
    screenshots: List[ScreenshotInfo] = []
    
    if os.path.exists(settings.SCREENSHOTS_DIR):
        for filename in os.listdir(settings.SCREENSHOTS_DIR):
            if filename.startswith(f"{task_id}_") and filename.endswith(".jpg"):
                screenshot_path = os.path.join(settings.SCREENSHOTS_DIR, filename)
                screenshot_url = f"/api/screenshots/{filename}"
                
                parts = filename.split("_")
                frame_number = 0
                track_id = None
                
                for i, part in enumerate(parts):
                    if part.startswith("frame") and i + 1 < len(parts):
                        try:
                            frame_number = int(part[5:])
                        except ValueError:
                            pass
                    if part.startswith("track") and i + 1 < len(parts):
                        try:
                            track_id = int(part[5:])
                        except ValueError:
                            pass
                
                stat = os.stat(screenshot_path)
                from datetime import datetime
                timestamp = datetime.fromtimestamp(stat.st_mtime).isoformat()
                
                screenshots.append(ScreenshotInfo(
                    frame_number=frame_number,
                    track_id=track_id,
                    screenshot_path=screenshot_path,
                    screenshot_url=screenshot_url,
                    timestamp=timestamp
                ))
    
    screenshots.sort(key=lambda x: x.frame_number)
    
    return TaskMediaResponse(
        task_id=task_id,
        video=video_info,
        screenshots=screenshots
    )
