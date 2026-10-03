"""Hand-written reference agents (numba). All take (st, body, occ, u, areas, scratch...)."""
import numpy as np
from numba import njit

from src.snake.engine import (S_FOOD, S_G, S_LEN, S_DIR, DR, DC, seg, move_outcome,
                              abs_dir, target_cell, flood_after_action, rng_next)


@njit(cache=True)
def _food_dist_after(st, body, a):
    G = st[S_G]
    cell = target_cell(st, body, abs_dir(st, a))
    f = st[S_FOOD]
    if cell < 0 or f < 0:
        return 1 << 30
    return abs(cell // G - f // G) + abs(cell % G - f % G)


@njit(cache=True)
def greedy_action(st, body, occ):
    """Greedy food following with a one-step safety filter (ties -> straight)."""
    best = 1
    best_d = 1 << 40
    for a in (1, 0, 2):
        if move_outcome(st, body, occ, a) != 0:
            continue
        dd = _food_dist_after(st, body, a)
        if dd < best_d:
            best_d = dd
            best = a
    return best


@njit(cache=True)
def floodfill_action(st, body, occ, stack, seen):
    """Strong reference: among safe moves keep those after which the tail is
    reachable (or the reachable area is at least the snake length); among these
    move greedily toward food. If none qualifies, take the move with the largest
    reachable area (ties -> tail-reachable, then straight)."""
    L = st[S_LEN]
    best = -1
    best_d = 1 << 40
    fb = 1
    fb_area = -2
    fb_tr = -1
    for a in (1, 0, 2):
        area, tr = flood_after_action(st, body, occ, a, stack, seen)
        if area < 0:
            continue
        if area > fb_area or (area == fb_area and tr > fb_tr):
            fb_area = area
            fb_tr = tr
            fb = a
        if tr == 1 or area >= L:
            dd = _food_dist_after(st, body, a)
            if dd < best_d:
                best_d = dd
                best = a
    if best >= 0:
        return best
    return fb


@njit(cache=True)
def random_action(rng_state):
    """Uniform over the 3 relative actions. Returns (action, new_state)."""
    x = rng_next(rng_state)
    return x % 3, x
