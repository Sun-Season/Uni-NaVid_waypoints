"""FastAPI server for robot navigation."""

import io
import time
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import SessionInfo, ActionResponse, HealthResponse, ErrorResponse
from .session_manager import SessionManager


# Global session manager (initialized in lifespan)
session_manager: SessionManager = None

# Configuration (set via environment or command line)
MODEL_PATH = None
LORA_PATH = None


def set_model_paths(model_path: str, lora_path: str = None):
    """Set model paths before starting the server."""
    global MODEL_PATH, LORA_PATH
    MODEL_PATH = model_path
    LORA_PATH = lora_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global session_manager

    print("=" * 50)
    print("Starting Navigation API Server")
    print("=" * 50)

    if MODEL_PATH is None:
        raise RuntimeError("Model path not set. Call set_model_paths() before starting.")

    print(f"Model path: {MODEL_PATH}")
    print(f"LoRA path: {LORA_PATH}")
    print("Loading model... (this may take a while)")

    session_manager = SessionManager(
        model_path=MODEL_PATH,
        lora_path=LORA_PATH,
        max_sessions=10,
    )

    # Load model once at startup
    session_manager.load_model()

    print("Server ready!")
    print("=" * 50)

    yield

    print("Shutting down server...")


app = FastAPI(
    title="Uni-NaVid Navigation API",
    description="Robot visual navigation cloud service",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        active_sessions=session_manager.active_session_count if session_manager else 0,
        model_loaded=session_manager.model_loaded if session_manager else False,
    )


@app.post("/api/v1/sessions", response_model=SessionInfo)
async def create_session(instruction: str = Form(None)):
    """
    Create a new navigation session.

    Each robot should create a session at the start of a navigation task.
    The session maintains visual feature cache for history observations.
    """
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    try:
        session_id = session_manager.create_session(instruction)
        session = session_manager.get_session(session_id)

        return SessionInfo(
            session_id=session_id,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session.created_at)),
            frame_count=0,
            instruction=instruction,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """End a navigation session and release resources."""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    if session_manager.remove_session(session_id):
        return {"message": "Session deleted", "session_id": session_id}

    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@app.post("/api/v1/navigate", response_model=ActionResponse)
async def navigate(
    session_id: str = Form(..., description="Session ID"),
    instruction: str = Form(..., description="Navigation instruction"),
    image: UploadFile = File(..., description="RGB image (JPEG/PNG)"),
):
    """
    Core navigation endpoint.

    Robot uploads current observation image, server returns predicted actions.

    Workflow:
    1. Validate session exists
    2. Decode image
    3. Call evaluator.get_action() for inference
    4. Return predicted actions
    """
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    # Validate session
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found. Create a session first."
        )

    try:
        # Read and decode image
        img_bytes = await image.read()
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))

        # Get action prediction
        start_time = time.time()
        action, did_inference, raw_output, step = session_manager.navigate(
            session_id, img, instruction
        )
        inference_time = (time.time() - start_time) * 1000

        return ActionResponse(
            session_id=session_id,
            step=step,
            actions=[action],
            did_inference=did_inference,
            raw_output=raw_output if did_inference else None,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/api/v1/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get information about a session."""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return SessionInfo(
        session_id=session_id,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session.created_at)),
        frame_count=session.frame_count,
        instruction=session.instruction,
    )
