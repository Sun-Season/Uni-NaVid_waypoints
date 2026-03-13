#!/usr/bin/env python3
"""
Mock server for testing API without loading the actual model.
"""

import io
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from server.models import SessionInfo, ActionResponse, HealthResponse


@dataclass
class MockSession:
    session_id: str
    instruction: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    frame_count: int = 0
    step: int = 0


# Mock session storage
sessions: Dict[str, MockSession] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Mock server started (no model loaded)")
    yield
    print("Mock server stopped")


app = FastAPI(
    title="Uni-NaVid Navigation API (Mock)",
    description="Mock server for testing",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        active_sessions=len(sessions),
        model_loaded=False,  # Mock mode
    )


@app.post("/api/v1/sessions", response_model=SessionInfo)
async def create_session(instruction: str = Form(None)):
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = MockSession(
        session_id=session_id,
        instruction=instruction,
    )
    return SessionInfo(
        session_id=session_id,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        frame_count=0,
        instruction=instruction,
    )


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/v1/navigate", response_model=ActionResponse)
async def navigate(
    session_id: str = Form(...),
    instruction: str = Form(...),
    image: UploadFile = File(...),
):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # Read image to verify it's valid
    img_bytes = await image.read()
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
    print(f"Received image: {img.shape}")

    session.frame_count += 1
    session.step += 1

    # Return mock actions
    mock_actions = ["forward", "forward", "left", "forward"]
    return ActionResponse(
        session_id=session_id,
        step=session.step,
        actions=[mock_actions[session.step % len(mock_actions)]],
        did_inference=True,
        raw_output="forward forward left forward (mock)",
    )


@app.get("/api/v1/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session.created_at)),
        frame_count=session.frame_count,
        instruction=session.instruction,
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
