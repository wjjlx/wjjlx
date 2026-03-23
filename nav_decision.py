"""
Navigation Decision Layer.

Reads the SemanticCognitiveMap and decides the next move.
Primary decision-maker is an LLM; rule-based fallback is always available.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Any

import numpy as np

from config import config
from vlm_mapper import SemanticCognitiveMap

logger = logging.getLogger(__name__)

# ── Cognitive states ──────────────────────────────────────────────────────────
STATE_EXPLORE = "EXPLORE"    # Blind exploration, no clear goal direction
STATE_ORIENT = "ORIENT"      # Goal direction known, heading towards it
STATE_APPROACH = "APPROACH"  # Goal is visible / very close

VALID_DIRS = ("up", "down", "left", "right")

_DIR_OFFSETS: dict[str, tuple[int, int]] = {
    "up":    (0, -1),
    "down":  (0,  1),
    "left":  (-1, 0),
    "right": (1,  0),
}

_DECISION_PROMPT = """You are the navigation cortex of a maze robot.
Read the cognitive map summary below and choose the BEST direction to move.

{map_summary}

Your last direction was: {last_direction}

Respond with ONLY a JSON object:
{{"direction": "up"|"down"|"left"|"right", "reasoning": "brief reason", "state": "EXPLORE"|"ORIENT"|"APPROACH"}}

Rules:
- Choose only a direction that is FREE (not wall/unknown) whenever possible.
- Prefer directions that bring you closer to the goal.
- Avoid immediately reversing (going opposite to last direction) unless there is no other option.
- state: APPROACH if goal is visible/adjacent; ORIENT if goal direction is known; EXPLORE otherwise.
"""


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", match.group())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


class NavigationDecider:
    """Decides the next navigation action given the current cognitive map."""

    def __init__(self) -> None:
        self._api_base = config.llm.api_base.rstrip("/")
        self._api_key = config.llm.api_key
        self._model = config.llm.llm_model
        self._timeout = config.llm.timeout
        self._cog_state: str = STATE_EXPLORE

    # ── Public API ────────────────────────────────────────────────────────────

    def decide(
        self,
        cog_map: SemanticCognitiveMap,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        last_direction: str = "",
    ) -> tuple[str, str]:
        """Return (direction, cognitive_state).

        Falls back to rule_based_fallback on any error.
        """
        try:
            direction, state = self._llm_decide(cog_map, agent_pos, goal_pos, last_direction)
            self._cog_state = state
            return direction, state
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM decision failed, using rule fallback: %s", exc)
            return self.rule_based_fallback(cog_map, agent_pos, goal_pos, last_direction)

    def rule_based_fallback(
        self,
        cog_map: SemanticCognitiveMap,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        last_direction: str = "",
    ) -> tuple[str, str]:
        """Simple rule-based navigation: goal-directed with obstacle avoidance."""
        ax, ay = agent_pos
        gx, gy = goal_pos

        # Determine preferred directions toward the goal
        preferred: list[str] = []
        if gx > ax:
            preferred.append("right")
        elif gx < ax:
            preferred.append("left")
        if gy > ay:
            preferred.append("down")
        elif gy < ay:
            preferred.append("up")

        # All four directions as fallback order
        all_dirs = list(preferred) + [d for d in VALID_DIRS if d not in preferred]

        # Remove reverse of last direction (avoid oscillation) unless forced
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
        opp = opposite.get(last_direction, "")
        non_reverse = [d for d in all_dirs if d != opp]
        candidates = non_reverse if non_reverse else all_dirs

        for direction in candidates:
            ddx, ddy = _DIR_OFFSETS[direction]
            nx, ny = ax + ddx, ay + ddy
            if 0 <= nx < cog_map.cols and 0 <= ny < cog_map.rows:
                cell = cog_map.occupancy[ny, nx]
                if cell != -1:          # not a known wall → try it
                    state = self._infer_state(agent_pos, goal_pos, direction, cog_map)
                    self._cog_state = state
                    return direction, state

        # Last resort: pick anything not a known wall
        for direction in VALID_DIRS:
            ddx, ddy = _DIR_OFFSETS[direction]
            nx, ny = ax + ddx, ay + ddy
            if 0 <= nx < cog_map.cols and 0 <= ny < cog_map.rows:
                if cog_map.occupancy[ny, nx] != -1:
                    return direction, STATE_EXPLORE

        return "up", STATE_EXPLORE  # should never reach here

    # ── Internal ──────────────────────────────────────────────────────────────

    def _llm_decide(
        self,
        cog_map: SemanticCognitiveMap,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        last_direction: str,
    ) -> tuple[str, str]:
        map_summary = cog_map.get_map_summary_for_llm(agent_pos, goal_pos)
        prompt = _DECISION_PROMPT.format(
            map_summary=map_summary,
            last_direction=last_direction or "none",
        )

        payload = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 128,
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
        data = _extract_json(raw_text)
        if not data:
            raise ValueError(f"Unparseable LLM response: {raw_text[:200]}")

        direction = str(data.get("direction", "")).lower()
        state = str(data.get("state", STATE_EXPLORE)).upper()

        if direction not in VALID_DIRS:
            raise ValueError(f"Invalid direction from LLM: {direction!r}")
        if state not in (STATE_EXPLORE, STATE_ORIENT, STATE_APPROACH):
            state = self._infer_state(agent_pos, goal_pos, direction, cog_map)

        return direction, state

    def _infer_state(
        self,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        direction: str,
        cog_map: SemanticCognitiveMap,
    ) -> str:
        ax, ay = agent_pos
        gx, gy = goal_pos
        dist = abs(gx - ax) + abs(gy - ay)
        if dist <= 2:
            return STATE_APPROACH
        # Check if goal direction is known from last perception
        if cog_map.perception_history:
            last = cog_map.perception_history[-1]
            if last.get("goal_visible"):
                return STATE_APPROACH
            if last.get("goal_direction"):
                return STATE_ORIENT
        return STATE_EXPLORE
