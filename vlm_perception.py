"""
VLM Perception Engine – "VLM as eyes".

Sends a compressed maze screenshot to a VLM and returns a PerceptionResult
describing the visible cells, area type, goal visibility and confidence.
Decision-making is NOT done here.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any

import numpy as np

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from config import config
from vlm_mapper import PerceptionResult

logger = logging.getLogger(__name__)

# Perception prompt template
_PERCEPTION_PROMPT = """You are the visual cortex of a maze-navigating robot.
The image shows the robot (blue circle) at position {agent_pos} navigating towards the goal (red square) at {goal_pos}.
The maze is {rows}×{cols} cells.

Carefully examine the image and respond with ONLY a JSON object (no markdown, no extra text):
{{
  "visible_cells": {{
    "up":    {{"row": {up_r},    "col": {up_c},    "state": "free"|"wall"}},
    "down":  {{"row": {dn_r},    "col": {dn_c},    "state": "free"|"wall"}},
    "left":  {{"row": {lt_r},    "col": {lt_c},    "state": "free"|"wall"}},
    "right": {{"row": {rt_r},    "col": {rt_c},    "state": "free"|"wall"}}
  }},
  "area_type": "corridor"|"junction"|"dead_end"|"open_area"|"unknown",
  "goal_visible": true|false,
  "goal_direction": "up"|"down"|"left"|"right"|"",
  "confidence": 0.0-1.0
}}

Rules:
- "state": "free" means the cell is passable path; "wall" means impassable.
- area_type "junction" means ≥3 exits; "dead_end" means only 1 exit; "corridor" means exactly 2 exits (straight or turn); "open_area" means large open space.
- goal_direction is the compass direction from the robot towards the goal (empty string if goal_visible is false).
- confidence: your self-assessed certainty (0=guessing, 1=certain).
"""


def _encode_image(image_array: np.ndarray, size: int = 320) -> str:
    """Resize image to *size*×*size* JPEG and return base64 string."""
    if not _PIL_AVAILABLE:
        # Fallback: encode raw array bytes (low quality but functional)
        raw = image_array.astype(np.uint8).tobytes()
        return base64.b64encode(raw).decode()

    img = PILImage.fromarray(image_array.astype(np.uint8))
    img = img.resize((size, size), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _extract_json(text: str) -> dict[str, Any] | None:
    """Robustly extract the first JSON object from *text*."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        # Try to fix common issues: trailing commas
        cleaned = re.sub(r",\s*([}\]])", r"\1", match.group())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


class VLMPerceptionEngine:
    """Calls the VLM to perceive the maze and returns a PerceptionResult."""

    def __init__(self) -> None:
        self._api_base = config.llm.api_base.rstrip("/")
        self._api_key = config.llm.api_key
        self._model = config.llm.vlm_model
        self._timeout = config.llm.timeout

    # ── Public API ────────────────────────────────────────────────────────

    def perceive(
        self,
        image_array: np.ndarray,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        maze_size: tuple[int, int],
    ) -> PerceptionResult:
        """Call VLM and return structured PerceptionResult.

        On any error an empty PerceptionResult is returned (never raises).
        """
        try:
            return self._perceive_inner(image_array, agent_pos, goal_pos, maze_size)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM perception failed: %s", exc)
            return PerceptionResult()

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        maze_size: tuple[int, int],
    ) -> str:
        ax, ay = agent_pos
        rows, cols = maze_size
        return _PERCEPTION_PROMPT.format(
            agent_pos=f"({ax},{ay})",
            goal_pos=f"({goal_pos[0]},{goal_pos[1]})",
            rows=rows,
            cols=cols,
            up_r=ay - 1, up_c=ax,
            dn_r=ay + 1, dn_c=ax,
            lt_r=ay,     lt_c=ax - 1,
            rt_r=ay,     rt_c=ax + 1,
        )

    def _perceive_inner(
        self,
        image_array: np.ndarray,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        maze_size: tuple[int, int],
    ) -> PerceptionResult:
        import urllib.request

        img_b64 = _encode_image(image_array, size=320)
        prompt = self._build_prompt(agent_pos, goal_pos, maze_size)

        payload = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": 512,
            }
        ).encode()

        req = urllib.request.Request(
            f"{self._api_base}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
            body = json.loads(resp.read().decode())

        raw_text = body["choices"][0]["message"]["content"]
        return self._parse_response(raw_text, agent_pos, maze_size)

    def _parse_response(
        self,
        raw: str,
        agent_pos: tuple[int, int],
        maze_size: tuple[int, int],
    ) -> PerceptionResult:
        data = _extract_json(raw)
        if not data:
            logger.warning("VLM returned unparseable JSON: %.200s", raw)
            return PerceptionResult(raw_response=raw)

        rows, cols = maze_size
        ax, ay = agent_pos

        # Parse visible_cells
        visible: dict[tuple[int, int], str] = {}
        vc_raw = data.get("visible_cells", {})
        if isinstance(vc_raw, dict):
            dir_offsets = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
            for dir_name, (dr, dc) in dir_offsets.items():
                nr, nc = ay + dr, ax + dc
                cell_data = vc_raw.get(dir_name)
                if isinstance(cell_data, dict):
                    state = str(cell_data.get("state", "")).lower()
                    if state in ("free", "wall"):
                        # Clamp to maze bounds
                        if 0 <= nr < rows and 0 <= nc < cols:
                            visible[(nr, nc)] = state

        area_type = str(data.get("area_type", "unknown")).lower()
        goal_visible = bool(data.get("goal_visible", False))
        goal_direction = str(data.get("goal_direction", "")).lower()
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return PerceptionResult(
            visible_cells=visible,
            area_type=area_type,
            goal_visible=goal_visible,
            goal_direction=goal_direction,
            confidence=confidence,
            raw_response=raw,
        )
