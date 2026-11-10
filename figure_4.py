#!/usr/bin/env python

import json
import subprocess as sbp
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# fix the number of samples for all tests
SAMPLES = 1e5

# run a single conditional Slatkin test
def run_test_conditional(
    n,
    k,
    configuration,
    initial_configuration=None,
    bias=1.0,
    samples=SAMPLES,
    seed=234,
):
    # build the call string for ewens-sampler
    if bias == 1.0:
        bstr = ""
    else:
        bstr = f"--bias {bias} "

    if initial_configuration is None:
        icstr = ""
    else:
        icstr = f"-i '{initial_configuration}' "

    cmd = f"ewens-sampler -j --seed {seed} {bstr}{icstr}-s {int(samples)} --test '{configuration}'"

    out = sbp.getoutput(cmd)

    # load output json and get the tail probabilities
    ans = json.loads(out)

    s = next(x for x in ans["tests"] if x["test"] == "slatkin")
    return s["p_lower"], s["p_upper"]

# run all tests at all bias parameter values
# for a single temporal window
def run(popes, s, size):
    # s is the start, e is the end of the window
    e = s + size

    # get popes up to the beginning of the window
    d = popes.query("start <= @s")

    # build initial frequency vector 
    year = d["start"].iloc[-1]
    pope = d["pope"].iloc[-1]
    initial_configuration = " ".join([str(x) for x in d["name"].value_counts()])
    n0 = d.shape[0]
    k0 = len(d["name"].unique())

    # get popes up to the end of the window
    d_end = popes.query("start < @e")
    # build end frequency vector
    end_configuration = " ".join([str(x) for x in d_end["name"].value_counts()])
    n = d_end.shape[0]
    k = len(d_end["name"].unique())

    # for each bias value, run a full exact test and a
    # conditional exact test
    rows = []
    for bias in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.11, 1.25, 1.43, 1.67, 2]:
        p_tail_win_lo, p_tail_win_up = run_test_conditional(
            n,
            k,
            end_configuration,
            initial_configuration,
            samples=SAMPLES,
            bias=bias,
            seed=int(n + k + bias * 1000),
        )
        p_tail_lo, p_tail_up = run_test_conditional(
            n,
            k,
            end_configuration,
            initial_configuration=None,
            samples=SAMPLES,
            bias=bias,
            seed=int(n + k + bias * 1000),
        )
        rows.append(
            {
                "year": year,
                "pope": pope,
                "n": n,
                "k": k,
                "window_start": s,
                "window_end": e,
                "window_size": size,
                "n0": n0,
                "k0": k0,
                "bias": bias,
                "p_tail_lower": p_tail_lo,
                "p_tail_upper": p_tail_up,
                "p_tail_win_lower": p_tail_win_lo,
                "p_tail_win_upper": p_tail_win_up,
            }
        )
    return pd.DataFrame.from_records(rows)

# read data
popes = pd.read_csv("popes-list.csv")

# reuse cache if present, recompute otherwise
if Path("./data/figure_4.csv").exists():
    print("using existing dataframe for figure 4")
    ps = pd.read_csv("./data/figure_4.csv")
else:
    # temporal window size
    size = 200
    inputs = []

    # get all window start dates (for each pope)
    for s in popes.query("start >= 432 and start < 2025 - @size + 5")["start"]:
        inputs.append((popes, s, size))

    # run all tests in parallel
    results = list(
        tqdm(
            Parallel(n_jobs=-1, return_as="generator")(
                delayed(run)(*y) for y in inputs
            ),
            total=len(inputs),
        )
    )
    ps = pd.concat(results)
    ps.to_csv("./data/figure_4.csv")

# plotting 
fig, axs = plt.subplots(
    1, 2, figsize=(12, 4), layout="compressed", sharex=True, sharey=True
)

ax = axs[0]

palette = sns.color_palette("icefire", n_colors=11)
for color, (bias, d) in zip(palette, ps.groupby("bias")):
    d = d.sort_values("window_end")
    ax.fill_between(
        d["window_end"],
        1 - d["p_tail_upper"],
        d["p_tail_lower"],
        color=color,
        alpha=1,
        linewidth=0.8,
        edgecolor=color,
    )
    ax.plot(d["window_end"], d["p_tail_lower"], color=color)
    ax.plot(d["window_end"], 1 - d["p_tail_upper"], color=color)

ax.set_xlabel("Year")
ax.set_ylabel("$P_E$, Slatkin exact test tail probability")
ax.axhline(0.025, color="red", alpha=0.6)
ax.axhline(0.975, color="red", alpha=0.6)
ax.axhline(1, color="gray", linestyle="--")
ax.axhline(0, color="gray", linestyle="--")

ax.text(
    -0.06,
    1.03,
    "A",
    transform=ax.transAxes,
    fontsize=16,
    fontweight="bold",
    va="baseline",
    ha="left",
)

ax = axs[1]

for color, (bias, d) in zip(palette, ps.groupby("bias")):
    d = d.sort_values("window_end")
    ax.fill_between(
        d["window_end"],
        1 - d["p_tail_win_upper"],
        d["p_tail_win_lower"],
        color=color,
        alpha=1,
        linewidth=0.8,
        edgecolor=color,
        label=f"{bias:g}",
    )

    ax.plot(d["window_end"], d["p_tail_win_lower"], color=color)
    ax.plot(d["window_end"], 1 - d["p_tail_win_upper"], color=color)

ax.set_xlabel("Year")
ax.set_ylabel("$P_E$, Slatkin exact test tail probability")
ax.axhline(0.025, color="red", alpha=0.6)
ax.axhline(0.975, color="red", alpha=0.6)
ax.axhline(1, color="gray", linestyle="--")
ax.axhline(0, color="gray", linestyle="--")

ax.legend(title="$\\beta$", loc="upper left", bbox_to_anchor=(1, 1))

ax.text(
    -0.06,
    1.03,
    "B",
    transform=ax.transAxes,
    fontsize=16,
    fontweight="bold",
    va="baseline",
    ha="left",
)

axs[0].set_box_aspect(2 / 3)
axs[1].set_box_aspect(2 / 3)

fig.savefig("figures/figure_4.pdf", bbox_inches="tight")
fig.savefig("figures/figure_4.png", bbox_inches="tight", dpi=600)
