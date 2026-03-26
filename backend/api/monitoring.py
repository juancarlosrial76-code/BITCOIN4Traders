from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health", tags=["monitoring"]) 
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# Prometheus metrics endpoint
from prometheus_fastapi_instrumentator import Instrumentator

instrumentor = Instrumentator()

def setup_metrics(app):
    instrumentor.instrument(app).expose(app)
