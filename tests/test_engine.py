import numpy as np
import pytest

from src.snake import engine as E
from src.baselines.heuristics import greedy_action, floodfill_action


def set_body(G, cells, direction, food, seed=1):
    """Build a state from an explicit body list (head first)."""
    st, body, occ = E.new_game(G, seed, 10)
    A = G * G
    body[:] = 0
    occ[:] = 0
    L = len(cells)
    for i, c in enumerate(reversed(cells)):
        body[i] = c
        occ[c] = 1
    st[E.S_HEAD] = L - 1
    st[E.S_LEN] = L
    st[E.S_DIR] = direction
    st[E.S_FOOD] = food
    return st, body, occ


def test_rng_deterministic_and_nonzero():
    a = E.rng_seed(42)
    b = E.rng_seed(42)
    assert a == b and 0 < a < 2**32
    seq = []
    x = a
    for _ in range(1000):
        x = E.rng_next(x)
        assert 0 < x < 2**32
        seq.append(x)
    assert len(set(seq)) == 1000


def test_new_game_layout():
    st, body, occ = E.new_game(10, 0, 1)
    assert st[E.S_LEN] == 3 and occ.sum() == 3
    assert E.seg(st, body, 0) == 5 * 10 + 5
    assert E.seg(st, body, 2) == 7 * 10 + 5
    assert st[E.S_FOOD] >= 0 and occ[st[E.S_FOOD]] == 0


def test_same_seed_same_food():
    s1, _, _ = E.new_game(10, 123, 1)
    s2, _, _ = E.new_game(10, 123, 1)
    s3, _, _ = E.new_game(10, 124, 1)
    assert s1[E.S_FOOD] == s2[E.S_FOOD]
    assert s1[E.S_RNG] != s3[E.S_RNG]


def test_wall_death():
    G = 5
    st, body, occ = set_body(G, [0 * G + 2, 1 * G + 2, 2 * G + 2], 0, 24)
    E.step(st, body, occ, 1)
    assert st[E.S_DONE] == 1 and st[E.S_CAUSE] == E.WALL


def test_body_death():
    G = 5
    # head at (2,2) heading left; turning right (up) -> hits body at (1,2)
    cells = [2 * G + 2, 2 * G + 3, 1 * G + 3, 1 * G + 2, 1 * G + 1]
    st, body, occ = set_body(G, cells, 3, 24)
    E.step(st, body, occ, 2)
    assert st[E.S_DONE] == 1 and st[E.S_CAUSE] == E.BODY


def test_moving_into_tail_is_allowed():
    G = 5
    # 2x2 loop: head (1,1) heading left? body square
    cells = [1 * G + 1, 1 * G + 2, 2 * G + 2, 2 * G + 1]  # tail at (2,1), below head
    st, body, occ = set_body(G, cells, 3, 24)  # heading left
    # turn left from heading left -> down -> into tail cell (2,1)
    E.step(st, body, occ, 0)
    assert st[E.S_DONE] == 0
    assert E.seg(st, body, 0) == 2 * G + 1
    assert occ.sum() == 4


def test_growth_and_food_respawn():
    G = 6
    st, body, occ = set_body(G, [2 * G + 2, 3 * G + 2, 4 * G + 2], 0, 1 * G + 2)
    ate = E.step(st, body, occ, 1)
    assert ate == 1 and st[E.S_LEN] == 4 and st[E.S_SCORE] == 1
    assert occ.sum() == 4
    assert st[E.S_FOOD] >= 0 and occ[st[E.S_FOOD]] == 0
    ate = E.step(st, body, occ, 1)
    assert ate == 0 and st[E.S_LEN] == 4 and occ.sum() == 4


def test_trapped_when_no_safe_move():
    G = 5
    # head (0,0) heading up at corner, body blocks right
    cells = [0, 1, 6, 5, 10]
    st, body, occ = set_body(G, cells, 0, 24)
    for a in range(3):
        assert E.move_outcome(st, body, occ, a) != 0
    E.step(st, body, occ, 1)
    assert st[E.S_CAUSE] == E.TRAPPED


def test_timeout():
    G = 4
    st, body, occ = E.new_game(G, 5, 1)  # limit = 16 steps without food
    st[E.S_FOOD] = -1  # remove food so it can never be eaten
    # circle in a 2x2 loop is impossible for len 3 without dying; use tail-chasing square
    moves = 0
    while not st[E.S_DONE] and moves < 100:
        E.step(st, body, occ, 2 if moves % 2 == 0 else 2)
        moves += 1
    # the length-3 snake turning right every step cycles a 2x2 block safely
    assert st[E.S_CAUSE] == E.TIMEOUT
    assert st[E.S_STEPS] == 17


def test_win_when_board_full():
    G = 2
    # length 3 on 2x2 board, food on last free cell
    cells = [0, 2, 3]  # head (0,0), then (1,0), (1,1); free: 1
    st, body, occ = set_body(G, cells, 0, 1)
    E.step(st, body, occ, 2)  # heading up, turn right -> (0,1)
    assert st[E.S_DONE] == 1 and st[E.S_CAUSE] == E.WIN and st[E.S_LEN] == 4


def test_flood_fill_area_and_tail():
    G = 5
    st, body, occ = set_body(G, [2 * G + 2, 3 * G + 2, 4 * G + 2], 0, 24)
    area, tr = E.flood_after_action(st, body, occ, 1, np.zeros(25, np.int64), np.zeros(25, np.int8))
    # after moving up: occupied = (1,2),(2,2),(3,2) ; free = 22 cells, all reachable
    assert area == 22 and tr == 1
    # wall move
    st2, body2, occ2 = set_body(G, [0 * G + 2, 1 * G + 2, 2 * G + 2], 0, 24)
    area, tr = E.flood_after_action(st2, body2, occ2, 1, np.zeros(25, np.int64), np.zeros(25, np.int8))
    assert area == -1


def test_features_shape_and_ranges():
    G = 10
    st, body, occ = E.new_game(G, 7, 1)
    u = np.zeros(E.N_FEATURES)
    areas = np.zeros(3)
    E.compute_features(st, body, occ, u, np.zeros(100, np.int64), np.zeros(100, np.int8),
                       np.zeros(100, np.int64), areas)
    assert u.shape == (21,) and np.all(u >= 0) and np.all(u <= 1)
    assert u[20] == pytest.approx(0.03)
    assert u[5:8].sum() == 0  # nothing adjacent at start
    assert len(E.FEATURE_NAMES) == E.N_FEATURES


@pytest.mark.parametrize("policy", ["greedy", "flood"])
def test_heuristics_play_without_crashing(policy):
    G = 8
    stack = np.zeros(64, np.int64)
    seen = np.zeros(64, np.int8)
    for seed in range(20):
        st, body, occ = E.new_game(G, seed, 2)
        while not st[E.S_DONE]:
            a = greedy_action(st, body, occ) if policy == "greedy" else floodfill_action(st, body, occ, stack, seen)
            E.step(st, body, occ, a)
            assert occ.sum() == st[E.S_LEN]
        assert st[E.S_CAUSE] != E.ALIVE


def test_features_v2_known_state():
    G = 10
    # head (5,5) heading up, body below; food at (2,7): 3 ahead, 2 to the right
    st, body, occ = set_body(G, [55, 65, 75], 0, 27)
    u = np.zeros(E.N_FEATURES_MAX)
    E.compute_features_v2(st, body, occ, u, np.zeros(100, np.int64))
    assert u[0] == pytest.approx(3 / 9) and u[1] == pytest.approx(2 / 9)
    assert u[3] == 1 and u[5] == 1 and u[4] == 0 and u[6] == 0
    assert u[7:10].sum() == 0
    # rays: left wall at distance 6 (cols 4..0 free, wall at -1), straight: rows 4..0 free -> 6, right: cols 6..9 -> 5
    assert u[10] == pytest.approx(1 / 6) and u[11] == pytest.approx(1 / 6) and u[12] == pytest.approx(1 / 5)
    assert u[13] == 0 and u[14] == 0


def test_features_v2_heading_rotation():
    G = 10
    # heading right at (5,5), body to the left; food directly below (7,5): lateral right = down
    st, body, occ = set_body(G, [55, 54, 53], 1, 75)
    u = np.zeros(E.N_FEATURES_MAX)
    E.compute_features_v2(st, body, occ, u, np.zeros(100, np.int64))
    assert u[0] == 0 and u[1] == pytest.approx(2 / 9) and u[5] == 1
