"""Session manager for handling multiple robot navigation sessions."""

import uuid
import time
import threading
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SessionState:
    """State for a single navigation session (no model, just state)."""
    session_id: str
    instruction: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    frame_count: int = 0
    step: int = 0
    last_output: str = ""
    # Session-specific state
    rgb_list: list = field(default_factory=list)
    pending_actions: list = field(default_factory=list)


class SessionManager:
    """
    Manages multiple robot navigation sessions with a shared model.

    The model is loaded once at startup and shared across all sessions.
    Each session maintains its own state (frame history, pending actions).
    """

    def __init__(
        self,
        model_path: str,
        lora_path: str = None,
        max_sessions: int = 10,
        session_timeout: int = 3600,  # 1 hour timeout
    ):
        self.model_path = model_path
        self.lora_path = lora_path
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()  # For model inference

        # Shared model components (loaded once)
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._model_loaded = False

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def load_model(self):
        """Load model once at startup."""
        if self._model_loaded:
            return

        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from uninavid.mm_utils import get_model_name_from_path
        from uninavid.model.builder import load_pretrained_model

        print(f"[SessionManager] Loading model from {self.model_path}")
        if self.lora_path:
            print(f"[SessionManager] Loading LoRA from {self.lora_path}")

        model_name = get_model_name_from_path(self.model_path)
        if self.lora_path:
            self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(
                self.lora_path, self.model_path, model_name
            )
        else:
            self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(
                self.model_path, None, model_name
            )

        self._model.eval()
        self._model_loaded = True
        print("[SessionManager] Model loaded successfully")

    def create_session(self, instruction: str = None) -> str:
        """Create a new navigation session."""
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        with self._lock:
            # Clean up if at capacity
            if len(self.sessions) >= self.max_sessions:
                oldest = min(self.sessions.values(), key=lambda s: s.last_access)
                self._remove_session(oldest.session_id)

            session_id = str(uuid.uuid4())[:8]

            self.sessions[session_id] = SessionState(
                session_id=session_id,
                instruction=instruction,
            )

            print(f"[SessionManager] Created session {session_id}")
            return session_id

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get a session by ID."""
        session = self.sessions.get(session_id)
        if session:
            session.last_access = time.time()
        return session

    def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        with self._lock:
            return self._remove_session(session_id)

    def _remove_session(self, session_id: str) -> bool:
        """Internal method to remove a session (must hold lock)."""
        if session_id in self.sessions:
            print(f"[SessionManager] Removing session {session_id}")
            del self.sessions[session_id]
            return True
        return False

    def navigate(
        self,
        session_id: str,
        image: np.ndarray,
        instruction: str,
    ) -> Tuple[str, bool, str, int]:
        """
        Process a navigation request.

        Args:
            session_id: Session identifier
            image: RGB image as numpy array
            instruction: Navigation instruction

        Returns:
            Tuple of (action, did_inference, raw_output, step)
        """
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        # Add frame to session's history
        session.rgb_list.append(image)
        session.frame_count += 1

        # If we have pending actions, return next one
        if len(session.pending_actions) > 0:
            action = session.pending_actions.pop(0)
            return action, False, session.last_output, session.step

        # Need to run inference - use lock to serialize model access
        with self._inference_lock:
            output = self._predict(session, instruction)
            actions = self._parse_actions(output)

            session.step += 1
            session.last_output = output
            session.pending_actions = []  # No caching, inference every request

            return actions[0], True, output, session.step

    def _predict(self, session: SessionState, instruction: str) -> str:
        """Run model inference for a session."""
        import torch
        from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from uninavid.conversation import conv_templates, SeparatorStyle

        # Special tokens
        VIDEO_START = "<video_special>"
        VIDEO_END = "</video_special>"
        IMAGE_START = "<image_special>"
        IMAGE_END = "</image_special>"
        NAV_TOKEN = "[Navigation]"
        IMAGE_SEP = "<image_sep>"

        # Initialize model state for this inference
        self._model.config.run_type = "eval"
        self._model.get_model().initialize_online_inference_nav_feat_cache()
        self._model.get_model().new_frames = len(session.rgb_list)

        # Process images
        batch_image = np.asarray(session.rgb_list)
        video = self._image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values']
        video = video.half().cuda()
        imgs = [video]

        # Clear frame history after processing
        session.rgb_list = []

        # Build prompt
        prompt_template = (
            "This is a navigation video. The instruction is: {}\n"
            "Based on the visual observation and instruction, determine your next four actions. "
            "The predicted action should be one of the following: forward, left, right, or wait."
        )
        question = prompt_template.format(instruction)
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
        video_start = self._tokenizer(VIDEO_START, return_tensors="pt").input_ids[0][1:].cuda()
        video_end = self._tokenizer(VIDEO_END, return_tensors="pt").input_ids[0][1:].cuda()
        image_start = self._tokenizer(IMAGE_START, return_tensors="pt").input_ids[0][1:].cuda()
        image_end = self._tokenizer(IMAGE_END, return_tensors="pt").input_ids[0][1:].cuda()
        nav_token = self._tokenizer(NAV_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_sep = self._tokenizer(IMAGE_SEP, return_tensors="pt").input_ids[0][1:].cuda()

        token_prompt = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]

        new_list = []
        while indices.numel() > 0:
            idx = indices[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start)
            new_list.append(image_sep)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end)
            new_list.append(image_start)
            new_list.append(image_end)
            new_list.append(nav_token)
            token_prompt = token_prompt[idx + 1:]
            indices = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]

        if token_prompt.numel() > 0:
            new_list.append(token_prompt)

        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self._tokenizer, input_ids)

        with torch.inference_mode():
            self._model.update_prompt([[question]])
            output_ids = self._model.generate(
                input_ids,
                images=imgs,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        input_len = input_ids.shape[1]
        outputs = self._tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]

        return outputs.strip()

    def _parse_actions(self, output: str) -> list:
        """Parse model output to action list."""
        NUM_ACTIONS = 4
        valid_actions = {'forward', 'left', 'right', 'wait', 'stop'}
        actions = []
        for word in output.lower().split():
            if word in valid_actions:
                actions.append('wait' if word == 'stop' else word)

        if len(actions) == 0:
            actions = ['wait'] * NUM_ACTIONS
        elif len(actions) < NUM_ACTIONS:
            while len(actions) < NUM_ACTIONS:
                actions.append(actions[-1])
        else:
            actions = actions[:NUM_ACTIONS]

        return actions

    def _cleanup_loop(self):
        """Periodically clean up expired sessions."""
        while True:
            time.sleep(60)
            current_time = time.time()
            with self._lock:
                expired = [
                    sid for sid, s in self.sessions.items()
                    if current_time - s.last_access > self.session_timeout
                ]
                for sid in expired:
                    print(f"[SessionManager] Cleaning up expired session: {sid}")
                    self._remove_session(sid)

    @property
    def active_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self.sessions)

    @property
    def model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
