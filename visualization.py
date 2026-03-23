"""
Visualization module.

CognitiveVisualizer renders:
  - The fog-of-war world (shown to human observer)
  - The cognitive map overlay (occupancy + semantic)
  - A full dashboard saved as PNG when goal is reached
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BG = (30, 30, 30)
C_WALL = (60, 60, 60)
C_WALL_KNOWN = (40, 40, 40)
C_FREE = (220, 220, 220)
C_UNKNOWN = (100, 100, 100)
C_FOG = (50, 50, 50)
C_AGENT = (70, 130, 230)
C_GOAL = (230, 70, 70)
C_TRAJ = (100, 200, 150)
C_TOPO_NODE = (255, 200, 50)

# Semantic colours
_SEM_COLOURS = {
    0: C_UNKNOWN,         # unknown
    1: (180, 220, 180),   # corridor
    2: (230, 200, 100),   # junction
    3: (230, 150, 150),   # dead_end
    4: (150, 200, 240),   # open_area
}


class CognitiveVisualizer:
    """Pygame-based renderer for the dual-world maze display."""

    def __init__(
        self,
        maze: np.ndarray,
        cell_size: int = 48,
        screenshot_dir: str = "out/screenshots",
    ) -> None:
        self.maze = maze
        self.rows, self.cols = maze.shape
        self.cell_size = cell_size
        self.screenshot_dir = screenshot_dir

        self._panel_w = self.cols * cell_size
        self._panel_h = self.rows * cell_size
        self._status_h = 40
        self._total_w = self._panel_w * 2 + 10  # two panels side-by-side
        self._total_h = self._panel_h + self._status_h

    # ── Pygame surface rendering ───────────────────────────────────────────────

    def render(
        self,
        screen: "pygame.Surface",
        cog_map: Any,                  # SemanticCognitiveMap
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        cog_state: str = "",
        step: int = 0,
        auto_nav: bool = False,
        font: "pygame.font.Font | None" = None,
    ) -> None:
        if not _PYGAME_AVAILABLE:
            return

        screen.fill(C_BG)
        cs = self.cell_size

        ax, ay = agent_pos
        gx, gy = goal_pos

        # ── Left panel: fog-of-war world (human view) ──────────────────────────
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * cs
                y = r * cs
                occ = cog_map.occupancy[r, c]
                if occ == 0:
                    colour = C_FOG
                elif self.maze[r, c] == 1:
                    colour = C_WALL_KNOWN
                else:
                    sem = int(cog_map.semantic[r, c])
                    colour = _SEM_COLOURS.get(sem, C_FREE)
                pygame.draw.rect(screen, colour, (x, y, cs - 1, cs - 1))

        # Trajectory
        for tx, ty in cog_map.trajectory[-200:]:
            px = tx * cs + cs // 2
            py = ty * cs + cs // 2
            pygame.draw.circle(screen, C_TRAJ, (px, py), max(2, cs // 8))

        # Topo nodes
        for _, (nr, nc) in cog_map.topo_nodes.items():
            px = nc * cs + cs // 2
            py = nr * cs + cs // 2
            pygame.draw.circle(screen, C_TOPO_NODE, (px, py), max(3, cs // 6), 2)

        # Goal
        gx_px = gx * cs + cs // 4
        gy_px = gy * cs + cs // 4
        pygame.draw.rect(screen, C_GOAL, (gx_px, gy_px, cs // 2, cs // 2))

        # Agent
        apx = ax * cs + cs // 2
        apy = ay * cs + cs // 2
        pygame.draw.circle(screen, C_AGENT, (apx, apy), cs // 3)

        # ── Right panel: full true world ──────────────────────────────────────
        ox = self._panel_w + 10
        for r in range(self.rows):
            for c in range(self.cols):
                x = ox + c * cs
                y = r * cs
                colour = C_WALL if self.maze[r, c] == 1 else C_FREE
                pygame.draw.rect(screen, colour, (x, y, cs - 1, cs - 1))

        # Agent in right panel
        pygame.draw.circle(
            screen, C_AGENT, (ox + apx, apy), cs // 3
        )
        pygame.draw.rect(
            screen, C_GOAL, (ox + gx_px, gy_px, cs // 2, cs // 2)
        )

        # ── Status bar ─────────────────────────────────────────────────────────
        if font:
            stats = cog_map.get_exploration_stats()
            pct = 100 * (stats["free"] + stats["walls"]) // (self.rows * self.cols)
            nav_text = f"SPACE=toggle auto  |  step={step}  |  state={cog_state}  |  explored={pct}%"
            if auto_nav:
                nav_text = "[AUTO] " + nav_text
            surf = font.render(nav_text, True, (200, 200, 200))
            screen.blit(surf, (8, self._panel_h + 8))

            # Panel labels
            lbl_l = font.render("Fog-of-War (cognitive map)", True, (160, 160, 160))
            lbl_r = font.render("True World (VLM input)", True, (160, 160, 160))
            screen.blit(lbl_l, (4, 4))
            screen.blit(lbl_r, (ox + 4, 4))

    def get_true_world_surface(
        self,
        screen: "pygame.Surface",
    ) -> np.ndarray:
        """Capture the right-panel (true world) as an RGB array for the VLM."""
        if not _PYGAME_AVAILABLE:
            return np.zeros((self._panel_h, self._panel_w, 3), dtype=np.uint8)
        arr = pygame.surfarray.array3d(screen)
        ox = self._panel_w + 10
        # pygame surfarray is (width, height, 3) – transpose to (height, width, 3)
        region = arr[ox: ox + self._panel_w, : self._panel_h]
        return np.transpose(region, (1, 0, 2))

    # ── Dashboard screenshot ────────────────────────────────────────────────────

    def plot_full_dashboard(
        self,
        cog_map: Any,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        steps_taken: int,
        exp_map: np.ndarray | None = None,
        save_path: str | None = None,
    ) -> str | None:
        """Render a matplotlib dashboard and save to disk.

        Compatible adapter: *exp_map* (legacy) is ignored if cog_map is provided.
        Returns the saved file path, or None if matplotlib is unavailable.
        """
        if not _MPL_AVAILABLE:
            return None

        os.makedirs(self.screenshot_dir, exist_ok=True)
        if save_path is None:
            ts = int(time.time())
            save_path = os.path.join(self.screenshot_dir, f"goal_reached_{ts}.png")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"CogNav – Goal Reached!  Steps: {steps_taken}  |  Agent: {agent_pos}  Goal: {goal_pos}",
            fontsize=13,
        )

        occ = cog_map.occupancy
        sem = cog_map.semantic

        # Panel 1: occupancy map
        ax0 = axes[0]
        ax0.set_title("Occupancy Map")
        im0 = ax0.imshow(occ, cmap="RdYlGn", vmin=-1, vmax=1, origin="upper")
        ax0.plot(agent_pos[0], agent_pos[1], "bo", markersize=10, label="Agent")
        ax0.plot(goal_pos[0], goal_pos[1], "r*", markersize=14, label="Goal")
        plt.colorbar(im0, ax=ax0, fraction=0.046)
        ax0.legend(fontsize=7)
        _add_legend_patches(ax0, [
            (-1, "Wall (known)", "red"),
            (0,  "Unknown",      "yellow"),
            (1,  "Free (known)", "green"),
        ])

        # Panel 2: semantic map
        ax1 = axes[1]
        ax1.set_title("Semantic Map")
        sem_display = np.zeros((*sem.shape, 3), dtype=np.uint8)
        for code, colour in _SEM_COLOURS.items():
            mask = sem == code
            sem_display[mask] = colour
        ax1.imshow(sem_display, origin="upper")
        ax1.plot(agent_pos[0], agent_pos[1], "bo", markersize=10)
        ax1.plot(goal_pos[0], goal_pos[1], "r*", markersize=14)
        _add_legend_patches(ax1, [
            (1, "Corridor",  "#b4dcb4"),
            (2, "Junction",  "#e6c864"),
            (3, "Dead-end",  "#e69696"),
            (4, "Open area", "#96c8f0"),
        ])

        # Panel 3: topology + trajectory
        ax2 = axes[2]
        ax2.set_title("Topology & Trajectory")
        ax2.imshow(occ, cmap="Greys", vmin=-1, vmax=1, origin="upper", alpha=0.5)
        # Trajectory
        if cog_map.trajectory:
            xs = [p[0] for p in cog_map.trajectory]
            ys = [p[1] for p in cog_map.trajectory]
            ax2.plot(xs, ys, "-", color="#64c896", linewidth=1.5, alpha=0.8)
        # Topo edges
        for a_id, b_id in cog_map.topo_edges:
            if a_id in cog_map.topo_nodes and b_id in cog_map.topo_nodes:
                ar_pos = cog_map.topo_nodes[a_id]
                br_pos = cog_map.topo_nodes[b_id]
                ax2.plot(
                    [ar_pos[1], br_pos[1]],
                    [ar_pos[0], br_pos[0]],
                    "-", color="#ffc832", linewidth=1,
                )
        # Topo nodes
        for _, npos in cog_map.topo_nodes.items():
            ax2.plot(npos[1], npos[0], "s", color="#ffc832", markersize=6)
        ax2.plot(agent_pos[0], agent_pos[1], "bo", markersize=10)
        ax2.plot(goal_pos[0], goal_pos[1], "r*", markersize=14)

        stats = cog_map.get_exploration_stats()
        fig.text(
            0.5, 0.01,
            f"Explored: {stats['free']} free + {stats['walls']} walls | "
            f"Dead-ends: {stats['dead_ends']} | Junctions: {stats['junctions']}",
            ha="center", fontsize=9,
        )

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return save_path


def _add_legend_patches(ax: Any, items: list) -> None:
    """Add small colour legend patches to a matplotlib axis."""
    if not _MPL_AVAILABLE:
        return
    patches = [
        mpatches.Patch(color=colour if isinstance(colour, str) else
                       tuple(c / 255 for c in colour), label=label)
        for _, label, colour in items
    ]
    ax.legend(handles=patches, fontsize=6, loc="lower right")
