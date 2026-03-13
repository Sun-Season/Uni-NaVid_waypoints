#!/usr/bin/env python3
"""
Robot client for communicating with the navigation server.

This module runs on the robot side and handles:
- Creating/managing navigation sessions
- Uploading images to the server
- Receiving and parsing action commands

Usage:
    from client.robot_client import RobotClient

    client = RobotClient("http://120.x.x.x:8000")
    client.create_session("Go to the kitchen")

    while navigating:
        image = capture_image()  # Your camera capture function
        actions = client.navigate(image, "Go to the kitchen")
        execute_action(actions[0])  # Your robot control function

    client.close_session()
"""

import io
from typing import List, Optional

import requests
import numpy as np
from PIL import Image


class RobotClient:
    """HTTP client for robot navigation."""

    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize the robot client.

        Args:
            server_url: URL of the navigation server (e.g., "http://120.x.x.x:8000")
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self._http_session = requests.Session()

    def health_check(self) -> dict:
        """Check if the server is healthy."""
        resp = self._http_session.get(
            f"{self.server_url}/api/v1/health",
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def create_session(self, instruction: str = None) -> str:
        """
        Create a new navigation session.

        Args:
            instruction: Optional navigation instruction

        Returns:
            Session ID
        """
        data = {}
        if instruction:
            data['instruction'] = instruction

        resp = self._http_session.post(
            f"{self.server_url}/api/v1/sessions",
            data=data,
            timeout=self.timeout
        )
        resp.raise_for_status()

        result = resp.json()
        self.session_id = result['session_id']
        print(f"Created session: {self.session_id}")
        return self.session_id

    def navigate(self, image: np.ndarray, instruction: str) -> List[str]:
        """
        Send an image and get navigation actions.

        Args:
            image: RGB image as numpy array (H, W, 3)
            instruction: Navigation instruction

        Returns:
            List of actions (e.g., ["forward", "left", "right", "wait"])
        """
        if self.session_id is None:
            raise RuntimeError("No active session. Call create_session() first.")

        # Encode image as JPEG
        img_pil = Image.fromarray(image)
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG', quality=85)
        buf.seek(0)

        # Send request
        resp = self._http_session.post(
            f"{self.server_url}/api/v1/navigate",
            data={
                'session_id': self.session_id,
                'instruction': instruction,
            },
            files={
                'image': ('frame.jpg', buf.getvalue(), 'image/jpeg'),
            },
            timeout=self.timeout
        )
        resp.raise_for_status()

        result = resp.json()
        return result['actions']

    def navigate_with_details(self, image: np.ndarray, instruction: str) -> dict:
        """
        Send an image and get full navigation response.

        Args:
            image: RGB image as numpy array (H, W, 3)
            instruction: Navigation instruction

        Returns:
            Full response dict with actions, step, did_inference, raw_output
        """
        if self.session_id is None:
            raise RuntimeError("No active session. Call create_session() first.")

        # Encode image as JPEG
        img_pil = Image.fromarray(image)
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG', quality=85)
        buf.seek(0)

        # Send request
        resp = self._http_session.post(
            f"{self.server_url}/api/v1/navigate",
            data={
                'session_id': self.session_id,
                'instruction': instruction,
            },
            files={
                'image': ('frame.jpg', buf.getvalue(), 'image/jpeg'),
            },
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def get_session_info(self) -> dict:
        """Get information about the current session."""
        if self.session_id is None:
            raise RuntimeError("No active session.")

        resp = self._http_session.get(
            f"{self.server_url}/api/v1/sessions/{self.session_id}",
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def close_session(self):
        """Close the current navigation session."""
        if self.session_id is None:
            return

        try:
            resp = self._http_session.delete(
                f"{self.server_url}/api/v1/sessions/{self.session_id}",
                timeout=self.timeout
            )
            resp.raise_for_status()
            print(f"Closed session: {self.session_id}")
        except Exception as e:
            print(f"Warning: Failed to close session: {e}")
        finally:
            self.session_id = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        self.close_session()


# Example usage
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test robot client')
    parser.add_argument('--server', type=str, default='http://localhost:8000')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--instruction', type=str, default='Go to the kitchen')
    args = parser.parse_args()

    client = RobotClient(args.server)

    # Health check
    print("Checking server health...")
    health = client.health_check()
    print(f"Server status: {health}")

    if args.image:
        # Test navigation
        print(f"\nTesting navigation with image: {args.image}")
        img = np.array(Image.open(args.image).convert('RGB'))

        with client:
            client.create_session(args.instruction)
            result = client.navigate_with_details(img, args.instruction)
            print(f"Actions: {result['actions']}")
            print(f"Step: {result['step']}")
            print(f"Did inference: {result['did_inference']}")
            if result.get('raw_output'):
                print(f"Raw output: {result['raw_output']}")
