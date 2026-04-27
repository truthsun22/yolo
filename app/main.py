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


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    setup_logging()
    
    logger.info(f"=" * 50)
    logger.info(f"{settings.APP_NAME} 启动中...")
    logger.info(f"版本: {settings.APP_VERSION}")
    logger.info(f"调试模式: {settings.DEBUG}")
    logger.info(f"=" * 50)
    
    yield
    
    logger.info(f"{settings.APP_NAME} 已关闭")


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
