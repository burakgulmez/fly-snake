"""Write paper/numbers.tex: every number in the paper as a macro read from results/."""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(ROOT, "results")
st = pd.read_csv(os.path.join(R, "subcircuit_stats.csv")).set_index("name")
ins = json.load(open(os.path.join(R, "data_inspection.json")))
s = pd.read_csv(os.path.join(R, "v2", "summary_models.csv")).set_index("exp")
c = pd.read_csv(os.path.join(R, "v2", "comparisons.csv"))
c = c[c.metric == "mean_score"].set_index("other")
dc = pd.read_csv(os.path.join(R, "v2", "death_causes.csv"))
d = dc[(dc.phase == "all") & (dc.exp == "v2_main_connectome")].iloc[0]
v1 = pd.read_csv(os.path.join(R, "summary_models.csv")).set_index("exp")
les = pd.read_csv(os.path.join(R, "v2", "lesion.csv"))
runs = pd.read_csv(os.path.join(R, "v2", "summary_runs.csv"))
cr = runs[runs.exp == "v2_main_connectome"]
cfg = json.load(open(os.path.join(R, "runs", "v2_main_connectome", "seed0", "config.json")))


def F(x, dgt=1):
    return f"{x:.{dgt}f}"


def thousands(x):
    return f"{x:,}".replace(",", "{,}")


a = st.loc["A_vpn_dn_130"]
nt = json.loads(a.nt_counts_edges)
lesion_rows = []
for sd, gg in les.groupby("seed"):
    lesion_rows.append((gg[gg.condition == "intact"].mean_score.iloc[0],
                        gg[gg.condition == "top_activity_var"].mean_score.iloc[0],
                        gg[gg.condition == "random"].mean_score.mean()))
lesion_rows = np.array(lesion_rows)


def model(exp, key):
    return F(s.loc[exp, key])


m = dict(
    NNeuronsTotal=thousands(ins["n_neurons"]), NPairs=thousands(ins["n_pairs"]),
    Neurons=int(a.neurons), Edges=int(a.edges), Inputs=int(a.n_input), Outputs=int(a.n_output),
    Intermediates=int(a.neurons - a.n_input - a.n_output), ExcPct=F(100 * a.frac_exc_edges), Reciprocity=F(a.reciprocity, 2),
    NAch=nt.get("ACH", 0), NGaba=nt.get("GABA", 0), NGlut=nt.get("GLUT", 0),
    NMono=sum(nt.get(k, 0) for k in ["DA", "SER", "OCT"]),
    NParams=int(s.loc["v2_main_connectome", "n_params"]), NSeeds=int(s.loc["v2_main_connectome", "n_seeds"]),
    TestGames=cfg["test_games"], BudgetSteps=thousands(cfg["max_train_steps"]),
    TrainTime=F(s.loc["v2_main_connectome", "train_time_s_mean"], 0),
    ConnMean=model("v2_main_connectome", "mean_score_mean"), ConnSd=model("v2_main_connectome", "mean_score_sd"),
    ConnMin=F(cr.mean_score.min()), ConnMax=F(cr.mean_score.max()), ConnMaxLen=int(cr.max_length.max()),
    ConnStepsPerFood=model("v2_main_connectome", "steps_per_food_mean"),
    DegMean=model("v2_main_degree_shuffle", "mean_score_mean"), DegSd=model("v2_main_degree_shuffle", "mean_score_sd"),
    DegP=F(c.loc["v2_main_degree_shuffle", "p"], 2),
    ErMean=model("v2_main_erdos_renyi", "mean_score_mean"), ErSd=model("v2_main_erdos_renyi", "mean_score_sd"),
    ErP=F(c.loc["v2_main_erdos_renyi", "p"], 2),
    SignMean=model("v2_main_sign_shuffle", "mean_score_mean"), SignSd=model("v2_main_sign_shuffle", "mean_score_sd"),
    SignP=F(c.loc["v2_main_sign_shuffle", "p"], 2),
    NoRecMean=model("v2_main_no_recurrence", "mean_score_mean"), NoRecSd=model("v2_main_no_recurrence", "mean_score_sd"),
    NoRecP=F(c.loc["v2_main_no_recurrence", "p"], 2),
    NoRaysMean=model("v2_abl_no_rays", "mean_score_mean"), NoRaysSd=model("v2_abl_no_rays", "mean_score_sd"),
    ASeventyMean=model("v2_scale_A70", "mean_score_mean"), ATwoEightyMean=model("v2_scale_A280", "mean_score_mean"),
    NeuronsSeventy=int(st.loc["A_vpn_dn_70", "neurons"]), NeuronsTwoEighty=int(st.loc["A_vpn_dn_200", "neurons"]),
    GreedyMean=model("v2_base_greedy", "mean_score_mean"), FloodMean=model("v2_base_floodfill", "mean_score_mean"),
    RandMean=model("v2_base_random", "mean_score_mean"),
    DeathBody=F(100 * d.body), DeathTrapped=F(100 * d.trapped), DeathWall=F(100 * d.wall), DeathTimeout=F(100 * d.timeout),
    LesTopBelow=int((lesion_rows[:, 1] < lesion_rows[:, 2]).sum()), LesSeeds=len(lesion_rows),
    LesK=int(les[les.condition == "top_activity_var"].n_silenced.iloc[0]),
    VOneMean=F(v1.loc["main_connectome", "mean_score_mean"]), VOneSd=F(v1.loc["main_connectome", "mean_score_sd"]),
    VOneNeurons=int(st.loc["A_vpn_dn_200", "neurons"]),
)
with open(os.path.join(ROOT, "paper", "numbers.tex"), "w") as fh:
    fh.write("% generated from results/ by paper/make_numbers.py -- do not edit\n")
    for k, v in m.items():
        fh.write("\\newcommand{\\" + k + "}{" + str(v) + "}\n")
print(open(os.path.join(ROOT, "paper", "numbers.tex")).read())
