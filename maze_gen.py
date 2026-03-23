"""
Maze generation using randomised Prim's algorithm.
Returns a 2-D numpy array where 0 = free cell, 1 = wall.
"""
import random
import numpy as np


def generate_maze(rows: int, cols: int, seed: int | None = None) -> np.ndarray:
    """Generate a perfect maze with odd dimensions.

    *rows* and *cols* should be odd numbers ≥ 5.
    Returns an int8 ndarray (rows × cols):  1 = wall, 0 = free.
    """
    if rows % 2 == 0:
        rows += 1
    if cols % 2 == 0:
        cols += 1

    rng = random.Random(seed)
    maze = np.ones((rows, cols), dtype=np.int8)

    # Start carving from (1, 1)
    start_r, start_c = 1, 1
    maze[start_r, start_c] = 0

    # Frontier: list of (wall_r, wall_c, from_r, from_c)
    def add_frontiers(r: int, c: int) -> list:
        result = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 0 < nr < rows - 1 and 0 < nc < cols - 1 and maze[nr, nc] == 1:
                result.append((nr, nc, r, c))
        return result

    frontiers = add_frontiers(start_r, start_c)

    while frontiers:
        idx = rng.randrange(len(frontiers))
        fr, fc, pr, pc = frontiers.pop(idx)
        if maze[fr, fc] == 1:
            maze[fr, fc] = 0
            maze[(fr + pr) // 2, (fc + pc) // 2] = 0
            frontiers.extend(add_frontiers(fr, fc))

    # Ensure start/end are free
    maze[1, 1] = 0
    maze[rows - 2, cols - 2] = 0

    return maze
