from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings, ensure_directories
from app.logger import setup_logging, get_logger
from app.routers import auth, tasks, events
from app.device_manager import device_manager
from app.video_source_manager import video_source_manager
from app.inference_engine import inference_engine


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    setup_logging()
    
    logger.info(f"=" * 50)
    logger.info(f"{settings.APP_NAME} 启动中...")
    logger.info(f"版本: {settings.APP_VERSION}")
    logger.info(f"调试模式: {settings.DEBUG}")
    
    logger.info(f"设备信息:")
    for device in device_manager.get_available_devices():
        logger.info(f"  - {device.device_type.value}: {device.name}")
    logger.info(f"首选设备: {device_manager.get_device_string(force_cpu=settings.FORCE_CPU)}")
    
    if not settings.FORCE_CPU and device_manager.is_gpu_available():
        logger.info("GPU加速已启用")
    else:
        logger.info("使用CPU模式运行")
    
    inference_engine.start()
    logger.info(f"推理引擎已启动: 批处理大小={inference_engine._batch_size}")
    
    logger.info(f"=" * 50)
    
    yield
    
    logger.info(f"=" * 50)
    logger.info(f"{settings.APP_NAME} 关闭中...")
    
    inference_engine.stop()
    logger.info("推理引擎已停止")
    
    video_source_manager.stop_all()
    logger.info("视频源管理器已停止")
    
    device_manager.shutdown()
    logger.info("设备管理器已关闭")
    
    logger.info(f"{settings.APP_NAME} 已关闭")
    logger.info(f"=" * 50)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(events.router)

app.mount("/api/videos", StaticFiles(directory=settings.VIDEO_OUTPUT_DIR), name="videos")
app.mount("/api/screenshots", StaticFiles(directory=settings.SCREENSHOTS_DIR), name="screenshots")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


@app.get("/api/info")
async def get_app_info():
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "default_model": settings.DEFAULT_MODEL,
        "available_models": settings.AVAILABLE_MODELS,
        "default_frame_skip": settings.DEFAULT_FRAME_SKIP,
        "event_cooldown_seconds": settings.EVENT_COOLDOWN_SECONDS,
        "log_retention_days": settings.LOG_RETENTION_DAYS
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
