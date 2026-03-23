"""
Three-layer Cognitive Map:
  - Occupancy layer  (int8):  0=unknown, 1=free, -1=wall
  - Semantic layer   (int8):  0=unknown, 1=corridor, 2=junction, 3=dead_end, 4=open_area
  - Topology layer:  nodes (dict pos->id) + edges (list of (id,id))

Also stores trajectory, step counter and perception history.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ──────────────────────────────────────────────
# Semantic area-type constants
# ──────────────────────────────────────────────
AREA_UNKNOWN = 0
AREA_CORRIDOR = 1
AREA_JUNCTION = 2
AREA_DEAD_END = 3
AREA_OPEN = 4

_AREA_NAMES = {
    AREA_UNKNOWN: "unknown",
    AREA_CORRIDOR: "corridor",
    AREA_JUNCTION: "junction",
    AREA_DEAD_END: "dead_end",
    AREA_OPEN: "open_area",
}
_AREA_FROM_STR = {v: k for k, v in _AREA_NAMES.items()}
TOPO_EDGE_MAX_DISTANCE = 4  # Manhattan distance threshold for auto-connecting topo nodes


# ──────────────────────────────────────────────
# PerceptionResult – output of VLM perception
# ──────────────────────────────────────────────
@dataclass
class PerceptionResult:
    """Structured output of a single VLM perception call."""

    # visible_cells: dict mapping (row, col) → 'free' | 'wall'
    visible_cells: dict[tuple[int, int], str] = field(default_factory=dict)
    area_type: str = "unknown"          # corridor / junction / dead_end / open_area
    goal_visible: bool = False
    goal_direction: str = ""            # up / down / left / right / ""
    confidence: float = 0.0            # 0-1
    raw_response: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.visible_cells and self.area_type == "unknown"


# ──────────────────────────────────────────────
# SemanticCognitiveMap
# ──────────────────────────────────────────────
class SemanticCognitiveMap:
    """Three-layer cognitive map updated incrementally from VLM perceptions."""

    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols

        # Layer 1: occupancy  –  0=unknown, 1=free, -1=wall
        self.occupancy: np.ndarray = np.zeros((rows, cols), dtype=np.int8)

        # Layer 2: semantic area type
        self.semantic: np.ndarray = np.zeros((rows, cols), dtype=np.int8)

        # Layer 3: topology
        self._next_node_id: int = 0
        self.topo_nodes: dict[int, tuple[int, int]] = {}       # id → (row, col)
        self._pos_to_node: dict[tuple[int, int], int] = {}     # (row,col) → id
        self.topo_edges: list[tuple[int, int]] = []            # (id, id)

        # Bookkeeping
        self.trajectory: list[tuple[int, int]] = []
        self.step: int = 0
        self.perception_history: list[dict[str, Any]] = []

    # ── Basic map updates ────────────────────────────────────────────────

    def mark_free(self, row: int, col: int) -> None:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.occupancy[row, col] = 1

    def mark_wall(self, row: int, col: int) -> None:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.occupancy[row, col] = -1

    # ── Update from perception ───────────────────────────────────────────

    def update_from_perception(
        self,
        result: PerceptionResult,
        agent_pos: tuple[int, int],
    ) -> None:
        """Merge a PerceptionResult into all three layers."""
        self.step += 1

        # Always mark current cell as free
        # agent_pos = (x, y) = (col, row)
        self.mark_free(agent_pos[1], agent_pos[0])

        # Update occupancy from visible_cells
        for (r, c), state in result.visible_cells.items():
            if state == "free":
                self.mark_free(r, c)
            elif state == "wall":
                self.mark_wall(r, c)

        # Update semantic layer at agent position
        area_code = _AREA_FROM_STR.get(result.area_type, AREA_UNKNOWN)
        agent_row, agent_col = agent_pos[1], agent_pos[0]
        if 0 <= agent_row < self.rows and 0 <= agent_col < self.cols:
            if area_code != AREA_UNKNOWN:
                self.semantic[agent_row, agent_col] = area_code

        # Update topology: add node if junction / open_area
        if area_code in (AREA_JUNCTION, AREA_OPEN):
            self._add_topo_node(agent_row, agent_col)

        # Extend trajectory
        self.trajectory.append(agent_pos)

        # Store in history
        self.perception_history.append(
            {
                "step": self.step,
                "agent_pos": agent_pos,
                "area_type": result.area_type,
                "goal_visible": result.goal_visible,
                "goal_direction": result.goal_direction,
                "confidence": result.confidence,
                "visible_count": len(result.visible_cells),
                "timestamp": time.time(),
            }
        )

    # ── Topology helpers ─────────────────────────────────────────────────

    def _add_topo_node(self, row: int, col: int) -> int:
        pos = (row, col)
        if pos in self._pos_to_node:
            return self._pos_to_node[pos]
        nid = self._next_node_id
        self._next_node_id += 1
        self.topo_nodes[nid] = pos
        self._pos_to_node[pos] = nid
        # Connect to nearby nodes
        for existing_id, existing_pos in self.topo_nodes.items():
            if existing_id != nid:
                er, ec = existing_pos
                dist = abs(row - er) + abs(col - ec)
                if dist <= TOPO_EDGE_MAX_DISTANCE:
                    self.topo_edges.append((nid, existing_id))
        return nid

    # ── Summary for LLM ─────────────────────────────────────────────────

    def get_map_summary_for_llm(
        self,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
    ) -> str:
        """Return a compact text description of the current map state."""
        ax, ay = agent_pos
        gx, gy = goal_pos

        # Direction to goal
        dx = gx - ax
        dy = gy - ay
        if abs(dx) > abs(dy):
            goal_dir = "right" if dx > 0 else "left"
        elif dy != 0:
            goal_dir = "down" if dy > 0 else "up"
        else:
            goal_dir = "reached"

        explored = int(np.sum(self.occupancy != 0))
        total = self.rows * self.cols
        pct = 100 * explored // total

        # Neighbours of agent
        neighbours: dict[str, str] = {}
        for name, (ddx, ddy) in {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }.items():
            nr, nc = ay + ddy, ax + ddx
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                v = self.occupancy[nr, nc]
                neighbours[name] = "free" if v == 1 else ("wall" if v == -1 else "unknown")
            else:
                neighbours[name] = "wall"

        area_name = _AREA_NAMES.get(int(self.semantic[ay, ax]), "unknown")
        stats = self.get_exploration_stats()

        return (
            f"Step {self.step}. Agent at ({ax},{ay}), Goal at ({gx},{gy}).\n"
            f"Goal direction: {goal_dir} (Δx={dx}, Δy={dy}).\n"
            f"Current area type: {area_name}.\n"
            f"Adjacent cells: up={neighbours['up']}, down={neighbours['down']}, "
            f"left={neighbours['left']}, right={neighbours['right']}.\n"
            f"Map explored: {pct}% ({explored}/{total} cells).\n"
            f"Dead-ends found: {stats['dead_ends']}, "
            f"Junctions found: {stats['junctions']}, "
            f"Topo nodes: {len(self.topo_nodes)}.\n"
        )

    # ── Exploration stats ────────────────────────────────────────────────

    def get_exploration_stats(self) -> dict[str, int]:
        free = int(np.sum(self.occupancy == 1))
        walls = int(np.sum(self.occupancy == -1))
        unknown = int(np.sum(self.occupancy == 0))
        dead_ends = int(np.sum(self.semantic == AREA_DEAD_END))
        junctions = int(np.sum(self.semantic == AREA_JUNCTION))
        return {
            "free": free,
            "walls": walls,
            "unknown": unknown,
            "dead_ends": dead_ends,
            "junctions": junctions,
        }

    # ── Persistence ──────────────────────────────────────────────────────

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "rows": self.rows,
            "cols": self.cols,
            "occupancy": self.occupancy.tolist(),
            "semantic": self.semantic.tolist(),
            "topo_nodes": {str(k): list(v) for k, v in self.topo_nodes.items()},
            "topo_edges": self.topo_edges,
            "trajectory": [list(p) for p in self.trajectory],
            "step": self.step,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "SemanticCognitiveMap":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        obj = cls(data["rows"], data["cols"])
        obj.occupancy = np.array(data["occupancy"], dtype=np.int8)
        obj.semantic = np.array(data["semantic"], dtype=np.int8)
        obj.topo_nodes = {int(k): tuple(v) for k, v in data["topo_nodes"].items()}
        obj._pos_to_node = {v: k for k, v in obj.topo_nodes.items()}
        obj._next_node_id = max(obj.topo_nodes.keys(), default=-1) + 1
        obj.topo_edges = [tuple(e) for e in data["topo_edges"]]
        obj.trajectory = [tuple(p) for p in data["trajectory"]]
        obj.step = data["step"]
        return obj
