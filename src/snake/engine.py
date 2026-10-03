"""Deterministic, numba-compiled Snake engine.

State of one game lives in three arrays so that many games can run inside
numba `prange` loops without Python objects:

  st   : int64[ST_SIZE]   scalar state (see S_* indices)
  body : int64[A]         circular buffer of cell indices, head at st[S_HEAD]
  occ  : int8[A]          1 if a body segment occupies the cell

Cells are indexed r*G + c. Absolute directions: 0 up, 1 right, 2 down, 3 left.
Relative actions: 0 turn left, 1 straight, 2 turn right.

The RNG is a 32-bit xorshift that is re-implemented bit-exactly in the JS demo.
"""
import numpy as np
from numba import njit

MASK32 = 0xFFFFFFFF

# scalar state indices
S_G, S_HEAD, S_LEN, S_DIR, S_FOOD, S_RNG, S_STEPS, S_SINCE, S_SCORE, S_DONE, S_CAUSE, S_TLIMIT = range(12)
ST_SIZE = 12

# death causes
ALIVE, WALL, BODY, TRAPPED, TIMEOUT, WIN = 0, 1, 2, 3, 4, 5
CAUSE_NAMES = ["alive", "wall", "body", "trapped", "timeout", "win"]

DR = np.array([-1, 0, 1, 0], dtype=np.int64)
DC = np.array([0, 1, 0, -1], dtype=np.int64)

INIT_LEN = 3


@njit(cache=True)
def rng_seed(seed):
    x = (seed * 2654435761 + 12345) & MASK32
    if x == 0:
        x = 1
    for _ in range(4):
        x = rng_next(x)
    return x


@njit(cache=True)
def rng_next(x):
    x ^= (x << 13) & MASK32
    x ^= x >> 17
    x ^= (x << 5) & MASK32
    return x & MASK32


@njit(cache=True)
def place_food(st, occ):
    A = st[S_G] * st[S_G]
    free = A - st[S_LEN]
    if free <= 0:
        st[S_FOOD] = -1
        return
    x = rng_next(st[S_RNG])
    st[S_RNG] = x
    r = x % free
    for cell in range(A):
        if occ[cell] == 0:
            if r == 0:
                st[S_FOOD] = cell
                return
            r -= 1


@njit(cache=True)
def new_game(G, seed, timeout_k):
    """Snake of length 3 in the centre, heading up, body extending downward."""
    A = G * G
    st = np.zeros(ST_SIZE, dtype=np.int64)
    body = np.zeros(A, dtype=np.int64)
    occ = np.zeros(A, dtype=np.int8)
    st[S_G] = G
    st[S_DIR] = 0
    st[S_RNG] = rng_seed(seed)
    st[S_TLIMIT] = timeout_k * A
    hr = G // 2
    hc = G // 2
    # buffer order: body[head], body[head-1], ... ; write tail first
    for i in range(INIT_LEN):
        cell = (hr + (INIT_LEN - 1 - i)) * G + hc
        body[i] = cell
        occ[cell] = 1
    st[S_HEAD] = INIT_LEN - 1
    st[S_LEN] = INIT_LEN
    place_food(st, occ)
    return st, body, occ


@njit(cache=True)
def seg(st, body, i):
    """Cell of body segment i (0 = head, len-1 = tail)."""
    A = body.shape[0]
    return body[(st[S_HEAD] - i + A) % A]


@njit(cache=True)
def abs_dir(st, action):
    return (st[S_DIR] + action - 1 + 4) % 4


@njit(cache=True)
def target_cell(st, body, d):
    """Cell one step from the head in absolute direction d, or -1 if outside."""
    G = st[S_G]
    h = seg(st, body, 0)
    r = h // G + DR[d]
    c = h % G + DC[d]
    if r < 0 or r >= G or c < 0 or c >= G:
        return -1
    return r * G + c


@njit(cache=True)
def move_outcome(st, body, occ, action):
    """0 = safe, WALL, or BODY for taking `action` from the current state."""
    d = abs_dir(st, action)
    cell = target_cell(st, body, d)
    if cell < 0:
        return WALL
    if occ[cell] == 1:
        tail = seg(st, body, st[S_LEN] - 1)
        eat = cell == st[S_FOOD]
        if not (cell == tail and not eat):
            return BODY
    return 0


@njit(cache=True)
def step(st, body, occ, action):
    """Advance one step. Returns 1 if food was eaten, else 0."""
    if st[S_DONE]:
        return 0
    A = st[S_G] * st[S_G]
    out = move_outcome(st, body, occ, action)
    st[S_STEPS] += 1
    if out != 0:
        forced = True
        for a in range(3):
            if move_outcome(st, body, occ, a) == 0:
                forced = False
                break
        st[S_DONE] = 1
        st[S_CAUSE] = TRAPPED if forced else out
        return 0
    d = abs_dir(st, action)
    cell = target_cell(st, body, d)
    eat = cell == st[S_FOOD]
    if not eat:
        tail = seg(st, body, st[S_LEN] - 1)
        occ[tail] = 0
    st[S_HEAD] = (st[S_HEAD] + 1) % A
    body[st[S_HEAD]] = cell
    occ[cell] = 1
    st[S_DIR] = d
    if eat:
        st[S_LEN] += 1
        st[S_SCORE] += 1
        st[S_SINCE] = 0
        if st[S_LEN] == A:
            st[S_DONE] = 1
            st[S_CAUSE] = WIN
            st[S_FOOD] = -1
            return 1
        place_food(st, occ)
        return 1
    st[S_SINCE] += 1
    if st[S_SINCE] > st[S_TLIMIT]:
        st[S_DONE] = 1
        st[S_CAUSE] = TIMEOUT
    return 0


# --------------------------------------------------------------------------
# Flood fill / space features
# --------------------------------------------------------------------------

@njit(cache=True)
def flood_after_action(st, body, occ, action, stack, seen):
    """Simulate `action`; flood fill free cells from the new head.

    Returns (area, tail_reachable). area counts reachable free cells (excluding
    the head). tail_reachable is 1 if the post-move tail cell is 4-adjacent to
    the head or to any reachable cell. Returns (-1, 0) if the move is fatal.
    `stack` (int64[A]) and `seen` (int8[A]) are scratch buffers.
    """
    if move_outcome(st, body, occ, action) != 0:
        return -1, 0
    G = st[S_G]
    A = G * G
    d = abs_dir(st, action)
    head = target_cell(st, body, d)
    eat = head == st[S_FOOD]
    L = st[S_LEN]
    old_tail = seg(st, body, L - 1)
    # new tail after move
    if eat:
        new_tail = old_tail
    else:
        new_tail = seg(st, body, L - 2)
    for i in range(A):
        seen[i] = occ[i]
    if not eat:
        seen[old_tail] = 0
    seen[head] = 1
    sp = 0
    stack[sp] = head
    sp += 1
    area = 0
    tail_reach = 0
    while sp > 0:
        sp -= 1
        cur = stack[sp]
        r = cur // G
        c = cur % G
        for k in range(4):
            rr = r + DR[k]
            cc = c + DC[k]
            if rr < 0 or rr >= G or cc < 0 or cc >= G:
                continue
            nb = rr * G + cc
            if nb == new_tail and nb != head:
                tail_reach = 1
            if seen[nb] == 0:
                seen[nb] = 1
                area += 1
                stack[sp] = nb
                sp += 1
    return area, tail_reach


# --------------------------------------------------------------------------
# Sensory features
# --------------------------------------------------------------------------

N_FEATURES = 21
FEATURE_NAMES = [
    "food_ahead", "food_behind", "food_left", "food_right", "food_dist",
    "danger1_L", "danger1_S", "danger1_R",
    "danger2_L", "danger2_S", "danger2_R",
    "danger3_L", "danger3_S", "danger3_R",
    "space_L", "space_S", "space_R",
    "tailreach_L", "tailreach_S", "tailreach_R",
    "fill",
]
# feature groups used for ablations
FEAT_FOOD = np.arange(0, 5)
FEAT_DANGER = np.arange(5, 14)
FEAT_SPACE = np.arange(14, 20)
FEAT_FILL = np.arange(20, 21)


@njit(cache=True)
def compute_features(st, body, occ, u, stack, seen, segidx, areas):
    """Fill u (float64[N_FEATURES]) and areas (float64[3], -1 = fatal)."""
    G = st[S_G]
    A = G * G
    L = st[S_LEN]
    h = seg(st, body, 0)
    hr = h // G
    hc = h % G
    d = st[S_DIR]
    # --- food in head frame
    for i in range(5):
        u[i] = 0.0
    f = st[S_FOOD]
    if f >= 0:
        vr = f // G - hr
        vc = f % G - hc
        fwd = vr * DR[d] + vc * DC[d]
        dr_ = (d + 1) % 4  # right-hand direction
        lat = vr * DR[dr_] + vc * DC[dr_]
        u[0] = 1.0 if fwd > 0 else 0.0
        u[1] = 1.0 if fwd < 0 else 0.0
        u[2] = 1.0 if lat < 0 else 0.0
        u[3] = 1.0 if lat > 0 else 0.0
        u[4] = (abs(vr) + abs(vc)) / (2.0 * (G - 1))
    # --- danger at distance 1..3 (segment i frees after L-i steps)
    for i in range(A):
        segidx[i] = -1
    for i in range(L):
        segidx[seg(st, body, i)] = i
    for a in range(3):
        ad = (d + a - 1 + 4) % 4
        blocked = 0
        for k in range(1, 4):
            rr = hr + DR[ad] * k
            cc = hc + DC[ad] * k
            if rr < 0 or rr >= G or cc < 0 or cc >= G:
                blocked = 1
            else:
                si = segidx[rr * G + cc]
                if si >= 0 and si < L - k:
                    blocked = 1
            if k == 1:
                # exact one-step danger uses the engine rule (tail/eating aware)
                blocked = 1 if move_outcome(st, body, occ, a) != 0 else 0
            u[5 + 3 * (k - 1) + a] = float(blocked)
    # --- space: reachable area / length (capped at 1) and tail reachability
    for a in range(3):
        area, tr = flood_after_action(st, body, occ, a, stack, seen)
        areas[a] = float(area)
        if area < 0:
            u[14 + a] = 0.0
            u[17 + a] = 0.0
        else:
            ratio = area / L
            u[14 + a] = ratio if ratio < 1.0 else 1.0
            u[17 + a] = float(tr)
    u[20] = L / A


# --------------------------------------------------------------------------
# Sensory features, version 2 (no look-ahead search): food vector + obstacle rays
# --------------------------------------------------------------------------

N_FEATURES_V2 = 15
FEATURE_NAMES_V2 = [
    "food_fwd", "food_lat", "food_dist", "food_ahead", "food_left", "food_right", "food_behind",
    "danger_L", "danger_S", "danger_R",
    "prox_L", "prox_S", "prox_R",
    "diag_FL", "diag_FR",
]
N_FEATURES_MAX = 21


@njit(cache=True)
def _blocked_at(st, segidx, r, c, k):
    """Wall, or body segment still present after k steps."""
    G = st[S_G]
    if r < 0 or r >= G or c < 0 or c >= G:
        return True
    si = segidx[r * G + c]
    return si >= 0 and si < st[S_LEN] - k


@njit(cache=True)
def compute_features_v2(st, body, occ, u, segidx):
    """Fill u[0:15].
    food_fwd/lat: signed food offset in the head frame (lat > 0 = right), / (G-1).
    danger_*: 1 if that action collides now (exact engine rule).
    prox_*: 1/d, d = distance (cells) to the first obstacle along the ray that is still
            present when reached (walls always); 0 if the ray is clear to the edge (cannot happen).
    diag_FL/FR: forward-left / forward-right cell is wall or body present after 1 step.
    """
    G = st[S_G]
    L = st[S_LEN]
    A = G * G
    h = seg(st, body, 0)
    hr = h // G
    hc = h % G
    d = st[S_DIR]
    for i in range(N_FEATURES_V2):
        u[i] = 0.0
    f = st[S_FOOD]
    if f >= 0:
        vr = f // G - hr
        vc = f % G - hc
        fwd = vr * DR[d] + vc * DC[d]
        rd = (d + 1) % 4
        lat = vr * DR[rd] + vc * DC[rd]
        u[0] = fwd / (G - 1.0)
        u[1] = lat / (G - 1.0)
        u[2] = (abs(vr) + abs(vc)) / (2.0 * (G - 1))
        u[3] = 1.0 if fwd > 0 else 0.0
        u[4] = 1.0 if lat < 0 else 0.0
        u[5] = 1.0 if lat > 0 else 0.0
        u[6] = 1.0 if fwd < 0 else 0.0
    for i in range(A):
        segidx[i] = -1
    for i in range(L):
        segidx[seg(st, body, i)] = i
    for a in range(3):
        u[7 + a] = 1.0 if move_outcome(st, body, occ, a) != 0 else 0.0
        ad = (d + a - 1 + 4) % 4
        for k in range(1, G + 1):
            if _blocked_at(st, segidx, hr + DR[ad] * k, hc + DC[ad] * k, k):
                u[10 + a] = 1.0 / k
                break
    ld = (d + 3) % 4
    rd2 = (d + 1) % 4
    fr_ = hr + DR[d]
    fc_ = hc + DC[d]
    u[13] = 1.0 if _blocked_at(st, segidx, fr_ + DR[ld], fc_ + DC[ld], 1) else 0.0
    u[14] = 1.0 if _blocked_at(st, segidx, fr_ + DR[rd2], fc_ + DC[rd2], 1) else 0.0
