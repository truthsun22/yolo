from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import get_current_active_user, User
from app.models import (
    TaskCreate, TaskResponse, TaskStatus, 
    Event, EventType
)
from app.task_manager import task_manager
from app.event_manager import EventManager
from app.logger import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix="/api/tasks", tags=["任务管理"])


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
