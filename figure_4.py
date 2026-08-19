#!/usr/bin/env python

import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import subprocess as sbp
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import delayed, Parallel

SAMPLES = 1e7


def run_test_conditional(
    n,
    k,
    configuration,
    initial_configuration=None,
    bias=1.0,
    samples=1e7,
    seed=234,
):
    if bias == 1.0:
        bstr = ""
    else:
        bstr = f"--bias {bias} "

    if initial_configuration is None:
        icstr = ""
    else:
        icstr = f"-i '{initial_configuration}' "

    cmd = f"ewinfer -j --seed {seed} {bstr}{icstr}-s {int(samples)} --test '{configuration}'"

    out = sbp.getoutput(cmd)
    # print(out)

    ans = json.loads(out)

    return [x for x in ans["tests"] if x["test"] == "slatkin"][0]["p_tail"]


def run(popes, s, size):
    e = s + size
    d = popes.query("start <= @s")

    year = d["start"].iloc[-1]
    pope = d["pope"].iloc[-1]
    initial_configuration = " ".join([str(x) for x in d["name"].value_counts()])
    n0 = d.shape[0]
    k0 = len(d["name"].unique())

    d_end = popes.query("start < @e")
    end_configuration = " ".join([str(x) for x in d_end["name"].value_counts()])
    n = d_end.shape[0]
    k = len(d_end["name"].unique())

    rows = []
    for bias in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.11, 1.25, 1.43, 1.67, 2]:
        p_tail_win = run_test_conditional(
            n,
            k,
            end_configuration,
            initial_configuration,
            samples=SAMPLES,
            bias=bias,
            seed=int(n + k + bias * 1000),
        )
        p_tail = run_test_conditional(
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
                "p_tail": p_tail,
                "p_tail_win": p_tail_win,
            }
        )
    return pd.DataFrame.from_records(rows)


popes = pd.read_csv("popes-list.csv")


if Path("./data/figure_4.csv").exists():
    print("using existing dataframe for figure 4")
    ps = pd.read_csv("./data/figure_4.csv")
else:
    size = 200
    inputs = []

    for s in popes.query("start >= 432 and start < 2025 - @size + 5")["start"]:
        inputs.append((popes, s, size))

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


fig, axs = plt.subplots(
    1, 2, figsize=(12, 4), layout="compressed", sharex=True, sharey=True
)

ax = axs[0]

sns.lineplot(
    ps,
    x="window_end",
    y="p_tail",
    hue="bias",
    ax=ax,
    palette=sns.color_palette("icefire", n_colors=11),
    legend=False,
)

ax.set_xlabel("Year")
ax.set_ylabel("$P_E$, Slatkin exact test tail probability")
ax.axhline(0.025, color="red", alpha=0.6)
ax.axhline(0.975, color="red", alpha=0.6)
ax.axhline(1, color="gray", linestyle="--")
ax.axhline(0, color="gray", linestyle="--")
# ax.set_title("Full configuration test")

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

sns.lineplot(
    ps,
    x="window_end",
    y="p_tail_win",
    hue="bias",
    ax=ax,
    palette=sns.color_palette("icefire", n_colors=11),
    legend=True,
)

ax.set_xlabel("Year")
ax.set_ylabel("$P_E$, Slatkin exact test tail probability")
ax.axhline(0.025, color="red", alpha=0.6)
ax.axhline(0.975, color="red", alpha=0.6)
ax.axhline(1, color="gray", linestyle="--")
ax.axhline(0, color="gray", linestyle="--")
# ax.set_title("Windowed conditional test")

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
