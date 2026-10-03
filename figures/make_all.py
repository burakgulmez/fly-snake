"""Figures for the Snake study (PDF + PNG), from results/v2 and results/runs/v2_*."""
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.brain.build import build_spec, load_circuit
from src.eval.rollout import trace_game
from src.snake import engine as E

R = os.path.join(ROOT, "results")
V2 = os.path.join(R, "v2")
FIG = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, ORANGE, RED, GRAY, LIGHT, INK2 = "#2a78d6", "#eb6834", "#e34948", "#8a8983", "#c9c8c2", "#52514e"
LABEL = {"v2_main_connectome": "FlyWire\ncircuit", "v2_main_degree_shuffle": "shuffled\n(same degrees)",
         "v2_main_erdos_renyi": "random\ngraph", "v2_main_sign_shuffle": "shuffled\nsigns",
         "v2_main_no_recurrence": "no internal\nconnections", "v2_base_greedy": "greedy\nrule",
         "v2_base_random": "random\nmoves"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_board(ax, st, body, G):
    """Snake as an outlined path in body order; triangle head points in the heading."""
    ax.set_xlim(-0.5, G - 0.5)
    ax.set_ylim(G - 0.5, -0.5)
    ax.set_aspect("equal")
    for i in range(G + 1):
        ax.axhline(i - 0.5, color="#e6e5e0", lw=0.4, zorder=0)
        ax.axvline(i - 0.5, color="#e6e5e0", lw=0.4, zorder=0)
    cells = [E.seg(st, body, i) for i in range(int(st[E.S_LEN]))]
    xs, ys = [c % G for c in cells], [c // G for c in cells]
    ax.plot(xs, ys, color="#0d366b", lw=6.3, solid_capstyle="round", solid_joinstyle="round", zorder=2)
    ax.plot(xs, ys, color=BLUE, lw=4.2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.plot(xs[0], ys[0], marker={0: "^", 1: ">", 2: "v", 3: "<"}[int(st[E.S_DIR])], ms=8, mfc="#0d366b",
            mec="white", mew=0.8, zorder=4)
    if st[E.S_FOOD] >= 0:
        ax.plot(st[E.S_FOOD] % G, st[E.S_FOOD] // G, "o", ms=5, color=RED, zorder=4)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_circuit():
    W, z = load_circuit("A_vpn_dn_130")
    n = W.shape[0]
    role = np.ones(n, int)
    role[z["inputs"]] = 0
    role[z["outputs"]] = 2
    order = np.lexsort((z["primary_type"], role))
    M = W.toarray()[order][:, order]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(np.sign(M), cmap=matplotlib.colors.ListedColormap([RED, "#ffffff", BLUE]), vmin=-1, vmax=1,
              interpolation="nearest")
    for c in np.cumsum([np.sum(role == r) for r in range(3)])[:-1]:
        ax.axhline(c - 0.5, color=INK2, lw=0.6)
        ax.axvline(c - 0.5, color=INK2, lw=0.6)
    ax.set_xlabel("presynaptic (VPN inputs | intermediate | DN outputs)")
    ax.set_ylabel("postsynaptic")
    ax.set_title(f"Visual-to-descending circuit: {n} neurons,\n{W.nnz} connections (blue excitatory, red inhibitory)")
    save(fig, "fig1_circuit")


def fig_training():
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for i, f in enumerate(sorted(glob.glob(os.path.join(R, "runs", "v2_main_connectome", "seed*", "train_log.csv")))):
        d = pd.read_csv(f).dropna(subset=["val_food"])
        ax.plot(d.total_steps / 1e6, d.val_food, lw=1.1, color=BLUE, alpha=0.5,
                label="10 training seeds" if i == 0 else None)
    ax.set_xlabel("training game steps (millions)")
    ax.set_ylabel("validation food per game\n(mean of 32 games)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig2_training")


def fig_comparison():
    runs = pd.read_csv(os.path.join(V2, "summary_runs.csv"))
    exps = [e for e in LABEL if e in set(runs.exp)]
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    for i, e in enumerate(exps):
        v = runs[runs.exp == e].mean_score.values
        col = BLUE if e == "v2_main_connectome" else (GRAY if e.startswith("v2_main") else LIGHT)
        ax.bar(i, v.mean(), width=0.62, color=col, edgecolor="white", lw=2)
        if len(v) > 1:
            ax.scatter(i + np.linspace(-0.2, 0.2, len(v)), v, s=8, color="#0b0b0b", zorder=3, lw=0)
        ax.text(i, v.max() + 1.2, f"{v.mean():.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(exps)))
    ax.set_xticklabels([LABEL[e] for e in exps], fontsize=8)
    ax.set_ylabel("food per test game")
    ax.set_ylim(0, 38)
    ax.set_title("500 unseen games per network; dots = training seeds; rules are deterministic", fontsize=9)
    save(fig, "fig3_comparison")


def fig_example_game():
    runs = pd.read_csv(os.path.join(V2, "summary_runs.csv"))
    cr = runs[runs.exp == "v2_main_connectome"].sort_values("mean_score").reset_index(drop=True)
    seed = int(cr.iloc[(len(cr) - 1) // 2].seed)  # median network
    run = os.path.join(R, "runs", "v2_main_connectome", f"seed{seed}")
    cfg = json.load(open(os.path.join(run, "config.json")))
    theta = np.load(os.path.join(run, "theta.npy"))
    spec, _, _ = build_spec(cfg, control_seed=seed)
    games = pd.read_csv(os.path.join(run, "test_games.csv"))
    gi = (games.score - games.score.median()).abs().idxmin()  # a typical game
    gseed = int(games.game_seed[gi])
    out, acts, _ = trace_game(spec, theta, gseed, cfg["grid"], cfg["timeout_k"])
    G = cfg["grid"]
    st, body, occ = E.new_game(G, gseed, cfg["timeout_k"])
    snaps = np.unique(np.linspace(0, len(acts) - 1, 6).astype(int))
    fig, axes = plt.subplots(1, len(snaps), figsize=(8, 1.9))
    k = 0
    for t in range(len(acts)):
        if t == snaps[k]:
            draw_board(axes[k], st, body, G)
            axes[k].set_title(f"step {t}\nlength {int(st[E.S_LEN])}", fontsize=7)
            k += 1
            if k == len(snaps):
                break
        E.step(st, body, occ, acts[t])
    fig.suptitle(f"A typical test game (network seed {seed}): {int(out[0])} food, ended by {E.CAUSE_NAMES[int(out[3])]}",
                 fontsize=8)
    save(fig, "fig4_example_game")


def fig_deaths():
    dc = pd.read_csv(os.path.join(V2, "death_causes.csv"))
    d = dc[(dc.phase == "all") & (dc.exp == "v2_main_connectome")].iloc[0]
    fig, ax = plt.subplots(figsize=(4.2, 1.4))
    left = 0
    for c, col, lab in [("body", ORANGE, "hit own body"), ("trapped", "#4a3aa7", "trapped (no safe move)"),
                        ("wall", RED, "wall"), ("timeout", "#eda100", "timeout")]:
        ax.barh(0, d[c], left=left, color=col, edgecolor="white", label=f"{lab} {100 * d[c]:.1f}%")
        left += d[c]
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("fraction of test games (all 10 networks)")
    ax.legend(frameon=False, fontsize=7, ncol=2, bbox_to_anchor=(0.5, -0.55), loc="upper center")
    save(fig, "fig5_deaths")


if __name__ == "__main__":
    for f in [fig_circuit, fig_training, fig_comparison, fig_example_game, fig_deaths]:
        f()
        print("ok", f.__name__)
