"""Pydantic data models for the navigation API."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    created_at: str
    frame_count: int = 0
    instruction: Optional[str] = None


class ActionResponse(BaseModel):
    """Navigation response with predicted actions."""
    session_id: str
    step: int = Field(..., description="Current step count")
    actions: List[str] = Field(..., description="Predicted actions: forward/left/right/wait")
    did_inference: bool = Field(..., description="Whether model inference was performed")
    raw_output: Optional[str] = Field(None, description="Raw model output text")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    active_sessions: int = 0
    model_loaded: bool = False


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
