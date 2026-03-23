"""
CogNav Maze – Main Entry Point.

Architecture:
  VLM (perception) → SemanticCognitiveMap → NavigationDecider → action

Dual-world rendering:
  - Left panel:  fog-of-war world (human view, cognitive map)
  - Right panel: true world (sent to VLM for perception)

Controls:
  SPACE   – toggle auto-navigation
  WASD / arrow keys – manual move
  R       – regenerate maze
  Q / ESC – quit
"""
from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from enum import Enum, auto

import numpy as np

# ── Try importing Pygame ───────────────────────────────────────────────────────
try:
    import pygame
    import pygame.font
except ImportError:
    print("pygame is required.  Install with:  pip install pygame", file=sys.stderr)
    sys.exit(1)

from config import config
from maze_gen import generate_maze
from nav_decision import NavigationDecider, _DIR_OFFSETS
from vlm_mapper import PerceptionResult, SemanticCognitiveMap
from vlm_perception import VLMPerceptionEngine
from visualization import CognitiveVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger("main")

# ── Pipeline step enum ─────────────────────────────────────────────────────────


class PipelineStep(Enum):
    STEP_PERCEIVE = auto()
    STEP_DECIDE = auto()


# ── Worker thread result types ─────────────────────────────────────────────────


class PerceiveResult:
    def __init__(self, result: PerceptionResult) -> None:
        self.result = result


class DecideResult:
    def __init__(self, direction: str, state: str) -> None:
        self.direction = direction
        self.state = state


# ── Main game ─────────────────────────────────────────────────────────────────


def build_new_game(rows: int, cols: int) -> tuple[np.ndarray, list[int], list[int], SemanticCognitiveMap]:
    """Generate maze and initial positions."""
    maze = generate_maze(rows, cols)
    agent_pos = [1, 1]
    goal_pos = [cols - 2, rows - 2]
    cog_map = SemanticCognitiveMap(rows, cols)
    # Mark start cell as free immediately
    cog_map.mark_free(agent_pos[1], agent_pos[0])
    return maze, agent_pos, goal_pos, cog_map


def move_agent(
    agent_pos: list[int],
    direction: str,
    maze: np.ndarray,
) -> bool:
    """Attempt to move agent; returns True on success."""
    ddx, ddy = _DIR_OFFSETS.get(direction, (0, 0))
    nx = agent_pos[0] + ddx
    ny = agent_pos[1] + ddy
    rows, cols = maze.shape
    if 0 <= nx < cols and 0 <= ny < rows and maze[ny, nx] == 0:
        agent_pos[0] = nx
        agent_pos[1] = ny
        return True
    return False


def _worker_perceive(
    image_array: np.ndarray,
    agent_pos: tuple[int, int],
    goal_pos: tuple[int, int],
    maze_size: tuple[int, int],
    engine: VLMPerceptionEngine,
    result_queue: "queue.Queue[PerceiveResult]",
) -> None:
    result = engine.perceive(image_array, agent_pos, goal_pos, maze_size)
    result_queue.put(PerceiveResult(result))


def _worker_decide(
    cog_map: SemanticCognitiveMap,
    agent_pos: tuple[int, int],
    goal_pos: tuple[int, int],
    last_direction: str,
    decider: NavigationDecider,
    result_queue: "queue.Queue[DecideResult]",
) -> None:
    direction, state = decider.decide(cog_map, agent_pos, goal_pos, last_direction)
    result_queue.put(DecideResult(direction, state))


def main() -> None:
    pygame.init()
    pygame.font.init()

    mc = config.maze
    rows, cols = mc.rows, mc.cols
    cell_size = mc.cell_size

    maze, agent_pos, goal_pos, cog_map = build_new_game(rows, cols)

    visualizer = CognitiveVisualizer(maze, cell_size, config.screenshot_dir)

    total_w = visualizer._total_w
    total_h = visualizer._total_h

    screen = pygame.display.set_mode((total_w, total_h))
    pygame.display.set_caption(config.window_title)

    try:
        font = pygame.font.SysFont("monospace", 14)
    except Exception:  # noqa: BLE001
        font = pygame.font.Font(None, 18)

    clock = pygame.time.Clock()

    # ── State ─────────────────────────────────────────────────────────────────
    auto_nav = False
    cog_state = "EXPLORE"
    last_direction = ""
    step = 0
    goal_reached = False
    screenshot_saved = False

    # Pipeline
    pipeline_step = PipelineStep.STEP_PERCEIVE
    worker_active = False
    result_queue: queue.Queue = queue.Queue()

    engine = VLMPerceptionEngine()
    decider = NavigationDecider()

    # Minimum delay between auto-nav steps (seconds)
    AUTO_STEP_INTERVAL = config.auto_step_interval
    last_auto_step = 0.0

    def trigger_perceive() -> None:
        nonlocal worker_active, pipeline_step
        if worker_active:
            return
        # Capture true-world image from right panel
        img = visualizer.get_true_world_surface(screen)
        pipeline_step = PipelineStep.STEP_PERCEIVE
        worker_active = True
        t = threading.Thread(
            target=_worker_perceive,
            args=(
                img,
                tuple(agent_pos),
                tuple(goal_pos),
                (rows, cols),
                engine,
                result_queue,
            ),
            daemon=True,
        )
        t.start()

    def trigger_decide() -> None:
        nonlocal worker_active, pipeline_step
        if worker_active:
            return
        pipeline_step = PipelineStep.STEP_DECIDE
        worker_active = True
        import copy
        snap = copy.deepcopy(cog_map)
        t = threading.Thread(
            target=_worker_decide,
            args=(
                snap,
                tuple(agent_pos),
                tuple(goal_pos),
                last_direction,
                decider,
                result_queue,
            ),
            daemon=True,
        )
        t.start()

    # ── Main loop ──────────────────────────────────────────────────────────────
    running = True
    while running:
        dt = clock.tick(config.fps)

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_r:
                    # Regenerate maze
                    maze, agent_pos, goal_pos, cog_map = build_new_game(rows, cols)
                    visualizer.maze = maze
                    cog_state = "EXPLORE"
                    last_direction = ""
                    step = 0
                    goal_reached = False
                    screenshot_saved = False
                    auto_nav = False
                    worker_active = False
                    pipeline_step = PipelineStep.STEP_PERCEIVE
                    with result_queue.mutex:
                        result_queue.queue.clear()

                elif event.key == pygame.K_SPACE:
                    auto_nav = not auto_nav
                    logger.info("Auto-nav: %s", "ON" if auto_nav else "OFF")
                    if auto_nav and not worker_active and not goal_reached:
                        trigger_perceive()

                # Manual movement
                elif not auto_nav:
                    dir_key_map = {
                        pygame.K_UP:    "up",
                        pygame.K_w:     "up",
                        pygame.K_DOWN:  "down",
                        pygame.K_s:     "down",
                        pygame.K_LEFT:  "left",
                        pygame.K_a:     "left",
                        pygame.K_RIGHT: "right",
                        pygame.K_d:     "right",
                    }
                    if event.key in dir_key_map:
                        direction = dir_key_map[event.key]
                        moved = move_agent(agent_pos, direction, maze)
                        if moved:
                            last_direction = direction
                            step += 1
                            # Minimal occupancy update for manual move
                            cog_map.mark_free(agent_pos[1], agent_pos[0])
                            cog_map.trajectory.append(tuple(agent_pos))
                            cog_map.step += 1

        # ── Poll result queue ──────────────────────────────────────────────────
        try:
            res = result_queue.get_nowait()
            worker_active = False

            if isinstance(res, PerceiveResult):
                # Update cognitive map with perception result
                cog_map.update_from_perception(res.result, tuple(agent_pos))
                logger.info(
                    "Perception done: %d cells | area=%s | goal_vis=%s",
                    len(res.result.visible_cells),
                    res.result.area_type,
                    res.result.goal_visible,
                )
                # Advance pipeline
                if auto_nav and not goal_reached:
                    trigger_decide()

            elif isinstance(res, DecideResult):
                direction = res.direction
                state = res.state
                cog_state = state

                moved = move_agent(agent_pos, direction, maze)
                if moved:
                    last_direction = direction
                    step += 1
                    logger.info("Moved %s (state=%s) → %s", direction, state, agent_pos)
                else:
                    # LLM suggested a wall – use rule fallback
                    logger.warning("Move %s blocked, using rule fallback", direction)
                    fb_dir, fb_state = decider.rule_based_fallback(
                        cog_map, tuple(agent_pos), tuple(goal_pos), last_direction
                    )
                    if move_agent(agent_pos, fb_dir, maze):
                        last_direction = fb_dir
                        cog_state = fb_state
                        step += 1

                # Update trajectory
                cog_map.trajectory.append(tuple(agent_pos))

                # Check goal
                if agent_pos[0] == goal_pos[0] and agent_pos[1] == goal_pos[1]:
                    goal_reached = True
                    auto_nav = False
                    logger.info("🎉 Goal reached in %d steps!", step)

                    if not screenshot_saved:
                        screenshot_saved = True
                        save_path = visualizer.plot_full_dashboard(
                            cog_map,
                            tuple(agent_pos),
                            tuple(goal_pos),
                            steps_taken=step,
                        )
                        if save_path:
                            logger.info("Dashboard saved → %s", save_path)

                elif auto_nav and not goal_reached:
                    # Continue pipeline: next perception
                    now = time.time()
                    if now - last_auto_step >= AUTO_STEP_INTERVAL:
                        last_auto_step = now
                        trigger_perceive()

        except queue.Empty:
            pass

        # ── Auto-nav: kick off first step if idle ──────────────────────────────
        if auto_nav and not worker_active and not goal_reached:
            now = time.time()
            if now - last_auto_step >= AUTO_STEP_INTERVAL:
                last_auto_step = now
                trigger_perceive()

        # ── Render ─────────────────────────────────────────────────────────────
        visualizer.maze = maze  # in case of reset
        visualizer.render(
            screen,
            cog_map,
            tuple(agent_pos),
            tuple(goal_pos),
            cog_state=cog_state,
            step=step,
            auto_nav=auto_nav,
            font=font,
        )

        if goal_reached:
            # Overlay victory message
            msg = font.render("🎉 GOAL REACHED!  Press R to restart.", True, (255, 230, 50))
            screen.blit(msg, (total_w // 2 - msg.get_width() // 2, total_h // 2))

        if worker_active:
            # Show thinking indicator
            dot_count = (int(time.time() * 2) % 4) + 1
            thinking = font.render("Thinking" + "." * dot_count, True, (100, 180, 255))
            screen.blit(thinking, (total_w - thinking.get_width() - 10, 8))

        pygame.display.flip()

    pygame.quit()
    logger.info("Exiting.")


if __name__ == "__main__":
    os.makedirs(config.screenshot_dir, exist_ok=True)
    main()
