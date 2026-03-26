"""
CantioAI Web User Interface Backend
FastAPI application providing REST API and WebSocket endpoints for the CantioAI web interface
"""

import asyncio
import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import librosa

# Import CantioAI components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.utils.optimization import create_optimized_model
from src.inference.realtime_engine import create_realtime_engine
from src.inference.synthesizer import synthesizer
from src.inference.vocoder import vocoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config("config.yaml")
webui_config = config.get('webui', {})

# Initialize FastAPI app
app = FastAPI(
    title="CantioAI Web Interface",
    description="Web interface for CantioAI voice conversion system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=webui_config.get('security', {}).get('cors_origins', ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize directories
UPLOAD_DIR = Path(webui_config.get('files', {}).get('upload_dir', "./uploads"))
RESULTS_DIR = Path("./results")
MODELS_DIR = Path("./models")

for directory in [UPLOAD_DIR, RESULTS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# Global state management
class AppState:
    def __init__(self):
        self.active_models: Dict[str, Any] = {}
        self.realtime_engines: Dict[str, Any] = {}
        self.processing_tasks: Dict[str, Dict] = {}
        self.websocket_connections: List[WebSocket] = []

app_state = AppState()

# Pydantic models for request/response validation
from pydantic import BaseModel

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    parameters: int
    size_mb: float
    loaded: bool

class AudioProcessRequest(BaseModel):
    audio_id: str
    model_id: str
    parameters: Dict[str, Any] = {}

class RealtimeConfig(BaseModel):
    model_id: str
    parameters: Dict[str, Any] = {}
    sample_rate: int = 24000
    chunk_size_ms: int = 50

class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0-100
    message: str
    result_url: Optional[str] = None
    error: Optional[str] = None

# API Routes

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "CantioAI Web Interface API", "version": "1.0.0"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Model Management APIs
@app.get("/api/models", response_model=List[ModelInfo])
async def list_models():
    """List available models"""
    # This would typically scan the models directory and return available models
    # For now, return a placeholder list
    models = [
        ModelInfo(
            id="base_model",
            name="CantioAI Base Model",
            description="Base hybrid source-filter + neural vocoder model",
            parameters=42000000,
            size_mb=160.5,
            loaded=True
        ),
        ModelInfo(
            id="optimized_model",
            name="CantioAI Optimized Model",
            description="Optimized model with quantization and pruning",
            parameters=35000000,
            size_mb=135.2,
            loaded=False
        )
    ]
    return models

@app.post("/api/models/{model_id}/load")
async def load_model(model_id: str, background_tasks: BackgroundTasks):
    """Load a specific model"""
    if model_id in app_state.active_models:
        return {"message": f"Model {model_id} already loaded"}

    # In a real implementation, this would load the actual model
    background_tasks.add_task(_load_model_background, model_id)
    return {"message": f"Loading model {model_id} in background"}

async def _load_model_background(model_id: str):
    """Background task to load model"""
    try:
        logger.info(f"Loading model {model_id}...")
        # Simulate model loading time
        await asyncio.sleep(2)
        # In reality, this would load the actual CantioAI model
        app_state.active_models[model_id] = {
            "id": model_id,
            "loaded_at": datetime.now(),
            "status": "ready"
        }
        logger.info(f"Model {model_id} loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")

@app.delete("/api/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a specific model"""
    if model_id in app_state.active_models:
        del app_state.active_models[model_id]
        # Also cleanup realtime engine if exists
        if model_id in app_state.realtime_engines:
            del app_state.realtime_engines[model_id]
        return {"message": f"Model {model_id} unloaded"}
    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

@app.get("/api/models/{model_id}/info", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """Get detailed information about a specific model"""
    # Placeholder implementation
    return ModelInfo(
        id=model_id,
        name=f"CantioAI {model_id.replace('_', ' ').title()}",
        description=f"Detailed information for model {model_id}",
        parameters=42000000,
        size_mb=160.5,
        loaded=model_id in app_state.active_models
    )

# Audio Processing APIs
@app.post("/api/audio/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file for processing"""
    # Validate file extension
    allowed_extensions = webui_config.get('files', {}).get('allowed_extensions', [".wav", ".mp3", ".flac", ".m4a"])
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File extension {file_ext} not allowed. Allowed: {allowed_extensions}"
        )

    # Validate file size
    max_size_mb = webui_config.get('files', {}).get('max_file_size', "100MB")
    max_size_bytes = int(max_size_mb.replace("MB", "")) * 1024 * 1024

    # Read file content to check size
    content = await file.read()
    if len(content) > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds limit of {max_size_mb}"
        )

    # Generate unique ID for the file
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    # Save file
    with open(file_path, "wb") as buffer:
        buffer.write(content)

    logger.info(f"Uploaded file {file.filename} as {file_id}")

    return {
        "file_id": file_id,
        "filename": file.filename,
        "size": len(content),
        "upload_time": datetime.now().isoformat(),
        "url": f"/api/audio/file/{file_id}"
    }

@app.get("/api/audio/file/{file_id}")
async def get_audio_file(file_id: str):
    """Retrieve uploaded audio file"""
    # Find file with matching ID
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.name.startswith(file_id + "_"):
            return FileResponse(file_path)

    raise HTTPException(status_code=404, detail="File not found")

@app.post("/api/audio/process")
async def process_audio(request: AudioProcessRequest, background_tasks: BackgroundTasks):
    """Process audio with specified model and parameters"""
    # Validate that file exists
    file_id = request.audio_id
    file_path = None
    for path in UPLOAD_DIR.iterdir():
        if path.name.startswith(file_id + "_"):
            file_path = path
            break

    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file {file_id} not found")

    # Validate that model is available/loaded
    model_id = request.model_id
    if model_id not in app_state.active_models:
        # Try to load it
        await _load_model_background(model_id)
        # Wait a bit for loading (in real implementation, this would be handled differently)
        await asyncio.sleep(1)

    # Create task
    task_id = str(uuid.uuid4())
    app_state.processing_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Task queued for processing",
        "file_id": file_id,
        "model_id": model_id,
        "parameters": request.parameters,
        "created_at": datetime.now(),
        "result_path": None
    }

    # Start background processing
    background_tasks.add_task(
        _process_audio_background,
        task_id,
        file_path,
        model_id,
        request.parameters
    )

    return {"task_id": task_id, "status": "queued"}

async def _process_audio_background(task_id: str, file_path: Path, model_id: str, parameters: Dict[str, Any]):
    """Background task to process audio"""
    try:
        # Update task status
        app_state.processing_tasks[task_id]["status"] = "processing"
        app_state.processing_tasks[task_id]["progress"] = 10.0
        app_state.processing_task[task_id]["message"] = "Loading audio..."

        # Load audio
        audio, sr = librosa.load(str(file_path), sr=None)

        # Update progress
        app_state.processing_tasks[task_id]["progress"] = 30.0
        app_state.processing_tasks[task_id]["message"] = "Processing with model..."

        # In a real implementation, this would use the actual CantioAI model
        # For now, simulate processing
        await asyncio.sleep(3)  # Simulate processing time

        # Update progress
        app_state.processing_tasks[task_id]["progress"] = 80.0
        app_state.processing_tasks[task_id]["message"] = "Generating output..."

        # Simulate result (in reality, this would be the processed audio)
        result_audio = audio  # Placeholder - just return original for now

        # Save result
        result_filename = f"result_{task_id}_{file_path.name}"
        result_path = RESULTS_DIR / result_filename

        # For demo purposes, just copy the file
        shutil.copy2(file_path, result_path)

        # Update task as completed
        app_state.processing_tasks[task_id]["status"] = "completed"
        app_state.processing_tasks[task_id]["progress"] = 100.0
        app_state.processing_tasks[task_id]["message"] = "Processing completed"
        app_state.processing_tasks[task_id]["result_path"] = str(result_path)
        app_state.processing_tasks[task_id]["completed_at"] = datetime.now()

        logger.info(f"Audio processing completed for task {task_id}")

    except Exception as e:
        logger.error(f"Error processing audio for task {task_id}: {e}")
        app_state.processing_tasks[task_id]["status"] = "failed"
        app_state.processing_tasks[task_id]["message"] = f"Processing failed: {str(e)}"
        app_state.processing_tasks[task_id]["error"] = str(e)

@app.get("/api/audio/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get status of processing task"""
    if task_id not in app_state.processing_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = app_state.processing_tasks[task_id]
    return TaskStatus(
        task_id=task["task_id"],
        status=task["status"],
        progress=task["progress"],
        message=task["message"],
        result_url=f"/api/audio/result/{task_id}" if task["status"] == "completed" and task.get("result_path") else None,
        error=task.get("error")
    )

@app.get("/api/audio/result/{task_id}")
async def get_processed_result(task_id: str):
    """Get processed audio result"""
    if task_id not in app_state.processing_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = app_state.processing_tasks[task_id]
    if task["status"] != "completed" or not task.get("result_path"):
        raise HTTPException(status_code=400, detail="Result not available")

    result_path = Path(task["result_path"])
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(result_path)

@app.delete("/api/audio/cancel/{task_id}")
async def cancel_task(task_id: str):
    """Cancel processing task"""
    if task_id not in app_state.processing_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = app_state.processing_tasks[task_id]
    if task["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed task")

    task["status"] = "cancelled"
    task["message"] = "Task cancelled by user"
    return {"message": f"Task {task_id} cancelled"}

# WebSocket for real-time processing
@app.websocket("/ws/realtime/{client_id}")
async def websocket_realtime(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time audio processing"""
    await websocket.accept()
    app_state.websocket_connections.append(websocket)
    logger.info(f"WebSocket client {client_id} connected")

    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message["type"] == "config":
                # Configure real-time processing
                config_data = message["config"]
                await _handle_realtime_config(websocket, client_id, config_data)
            elif message["type"] == "audio_chunk":
                # Process audio chunk
                await _handle_audio_chunk(websocket, client_id, message["data"])
            elif message["type"] == "ping":
                # Respond to ping
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")
        if websocket in app_state.websocket_connections:
            app_state.websocket_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        if websocket in app_state.websocket_connections:
            app_state.websocket_connections.remove(websocket)

async def _handle_realtime_config(websocket: WebSocket, client_id: str, config: Dict[str, Any]):
    """Handle real-time processing configuration"""
    model_id = config.get("model_id")
    parameters = config.get("parameters", {})

    try:
        # Create or get realtime engine for this model
        if model_id not in app_state.realtime_engines:
            # Load model if needed
            if model_id not in app_state.active_models:
                # In real implementation, load the model
                pass

            # Create realtime engine
            # This would integrate with the actual CantioAI realtime engine
            app_state.realtime_engines[model_id] = {
                "model_id": model_id,
                "parameters": parameters,
                "created_at": datetime.now()
            }

        await websocket.send_text(json.dumps({
            "type": "config_response",
            "status": "success",
            "message": f"Real-time processing configured for model {model_id}"
        }))

    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "config_response",
            "status": "error",
            "message": f"Configuration failed: {str(e)}"
        }))

async def _handle_audio_chunk(websocket: WebSocket, client_id: str, audio_data: List[float]):
    """Handle incoming audio chunk for real-time processing"""
    try:
        # Convert audio data to numpy array
        audio_np = np.array(audio_data, dtype=np.float32)

        # In a real implementation, this would:
        # 1. Send audio to the appropriate realtime engine
        # 2. Get processed audio back
        # 3. Send processed audio back to client

        # For now, just echo back the same audio (placeholder)
        await websocket.send_text(json.dumps({
            "type": "audio_result",
            "data": audio_np.tolist(),
            "timestamp": datetime.now().isoformat()
        }))

    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Audio processing error: {str(e)}"
        }))

# System management APIs
@app.get("/api/system/status")
async def get_system_status():
    """Get system status"""
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(app_state.active_models),
        "active_tasks": len([t for t in app_state.processing_tasks.values() if t["status"] == "processing"]),
        "websocket_connections": len(app_state.websocket_connections),
        "memory_usage_mb": 0,  # Would be calculated in real implementation
        "uptime_seconds": 0  # Would be calculated in real implementation
    }

@app.get("/api/system/resources")
async def get_system_resources():
    """Get system resource usage"""
    # Placeholder implementation
    return {
        "cpu_usage_percent": 0,
        "memory_usage_mb": 0,
        "gpu_usage_percent": 0 if not torch.cuda.is_available() else 0,
        "disk_usage_percent": 0,
        "active_processes": 0
    }

# Serve frontend static files (in production)
# This would be configured to serve the built React frontend
# app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=webui_config.get('server', {}).get('host', "0.0.0.0"),
        port=webui_config.get('server', {}).get('port', 7860),
        workers=webui_config.get('server', {}).get('workers', 1)
    )