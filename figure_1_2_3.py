#!/usr/bin/env python

import json
import random
import subprocess as sbp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

# fix the number of samples for all
# Slatkin tests
SAMPLES = 1e7

# fix random seed
random.seed(231)

# read input data
popes = pd.read_csv("popes-list.csv")

# extract p-values from a test result json
def slatkin_tails(ans):
    s = next(x for x in ans["tests"] if x["test"] == "slatkin")
    return {"p_lower": s["p_lower"], "p_upper": s["p_upper"]}


# make figure 1

# count popes and names seen at each year
rows = []
for i in range(1, popes.shape[0] + 1):
    di = popes.iloc[:i, :]

    rows.append(
        {"year": di["start"].iloc[-1], "n": i, "names": len(di["name"].unique())}
    )

dc = pd.DataFrame.from_records(rows).sort_values(by="year")

# count popes per name and the year name was last chosen
rows = []
for name in popes["name"].unique():
    dn = popes.query("name == @name")

    rows.append(
        {
            "name": dn["pope"].iloc[0],
            "count": dn.shape[0],
            "last_seen": dn["start"].max(),
        }
    )

dn = pd.DataFrame.from_records(rows)

fig, axs = plt.subplots(
    1, 2, figsize=(10, 4), constrained_layout=True, gridspec_kw={"wspace": 0.1}
)
colors = sns.xkcd_palette(["charcoal", "terracotta"])

ax = axs[0]
sns.lineplot(dc, x="year", y="n", label="Popes", color=colors[0], ax=ax)
sns.lineplot(dc, x="year", y="names", label="Distinct names", color=colors[1], ax=ax)

ax.set_xlabel("Year")
ax.set_ylabel("Cumulative count")
ax.set_xticks([1, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])

ax = axs[1]
sns.scatterplot(data=dn, x="count", y="last_seen", color=colors[0], alpha=0.0, ax=ax)

for _, row in dn.iterrows():
    if "John Paul" in row["name"]:
        letter = "JP"
    else:
        letter = row["name"].upper()[0]
    ax.text(
        row["count"],
        row["last_seen"],
        letter,
        color=colors[0],
        ha="center",
        va="center_baseline",
    )

ax.set_xlabel("Count of popes with the name")
ax.set_ylabel("Year the name was last chosen")

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter("{x:.0f}")
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.set_yticks([1, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])


ax.set_xlim(0, 23)
ax.set_ylim(-70, 2300)

axs[0].text(
    -0.1,
    1.1,
    "A",
    transform=axs[0].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="center",
)

axs[1].text(
    -0.1,
    1.1,
    "B",
    transform=axs[1].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="center",
)

ax.annotate(
    "John",
    (21.2, 1958 - 50),
    (22, 1958 - 250),
    arrowprops={
        "arrowstyle": "->",
        "connectionstyle": "arc3,rad=-0.05",
        "color": colors[1],
        "shrinkA": 0.1,
        "shrinkB": 0.8,
    },
    va="baseline",
    ha="center",
    color=colors[1],
)

ax.annotate(
    "Gregory",
    (16.2, 1831 - 50),
    (17, 1831 - 250),
    arrowprops={
        "arrowstyle": "->",
        "connectionstyle": "arc3,rad=-0.05",
        "color": colors[1],
        "shrinkA": 0.1,
        "shrinkB": 0.8,
    },
    va="baseline",
    ha="center",
    color=colors[1],
)

ax.annotate(
    "Benedict",
    (15.2, 2005 + 33),
    (16, 2005 + 150),
    arrowprops={
        "arrowstyle": "->",
        "connectionstyle": "arc3,rad=-0.05",
        "color": colors[1],
        "shrinkA": 0.1,
        "shrinkB": 0.8,
    },
    va="baseline",
    ha="left",
    color=colors[1],
)

ax.annotate(
    "Leo",
    (13.8, 2025 + 33),
    (13, 2025 + 150),
    arrowprops={
        "arrowstyle": "->",
        "connectionstyle": "arc3,rad=-0.05",
        "color": colors[1],
        "shrinkA": 0.1,
        "shrinkB": 0.8,
    },
    va="baseline",
    ha="right",
    color=colors[1],
)

ax.annotate(
    "Pius",
    (11.8, 1939 + 33),
    (11, 1939 + 150),
    arrowprops={
        "arrowstyle": "->",
        "connectionstyle": "arc3,rad=-0.05",
        "color": colors[1],
        "shrinkA": 0.1,
        "shrinkB": 0.8,
    },
    va="baseline",
    ha="right",
    color=colors[1],
)

fig.savefig("figures/figure_1.pdf", bbox_inches="tight")
fig.savefig("figures/figure_1.png", dpi=600, bbox_inches="tight")


# make figure 2

# try to use cached data or recompute from scratch
if Path("./data/figure_2.csv").exists():
    print("using existing dataframe for figure 2")
    ps = pd.read_csv("./data/figure_2.csv")
else:
    print("computing data for figure 2")

    ## make a seed for each pope starting from 44th:
    random.seed(231)
    args = [(i, random.randint(0, 2**16)) for i in range(44, popes.shape[0] + 1)]

    # run ewens sampler on the data up to pope number i
    def run(i, seed):
        # subset popes
        d = popes.iloc[:i, :]

        year = d["start"].iloc[-1]
        pope = d["pope"].iloc[-1]

        # get observed n (number of popes) and k (number of names)
        n, k = d.shape[0], len(d["name"].unique())

        # build a frequency vector from value counts
        configuration = " ".join([str(x) for x in d["name"].value_counts()])

        # run the test and read json output
        cmd = (
            f"ewens-sampler -j --seed {seed} -s {int(SAMPLES)} --test '{configuration}'"
        )
        ans = json.loads(sbp.getoutput(cmd))

        result = {
            "year": year,
            "pope": pope,
            "n": n,
            "k": k,
            "theta": ans["theta"],
        }

        # add p_values from the results
        result.update(slatkin_tails(ans))

        return result

    # run in parallel on data up to each pope
    rows = list(
        tqdm(
            Parallel(n_jobs=20, return_as="generator")(delayed(run)(*x) for x in args),
            total=len(args),
        )
    )

    # aggregate and cache result
    ps = pd.DataFrame.from_records(rows)
    ps.to_csv("./data/figure_2.csv")

# plot p-values and inferred theta 
fig, axs = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True, sharex=True)
colors = sns.xkcd_palette(["slate blue", "teal green"])

axs[0].fill_between(
    ps["year"],
    1 - ps["p_upper"],
    ps["p_lower"],
    color=colors[0],
    alpha=0.5,
    linewidth=0.8,
    edgecolor=colors[0],
)
axs[0].plot(ps["year"], ps["p_lower"], color=colors[0])
axs[0].plot(ps["year"], 1 - ps["p_upper"], color=colors[0])

axs[0].set_xlabel("Year")
axs[0].set_ylabel("$P_E$, Slatkin exact test tail probability", fontsize="small")

axs[0].axhline(1, color="gray", linestyle="--")
axs[0].axhline(0.025, color="red", alpha=0.6)
axs[0].axhline(0.975, color="red", alpha=0.6)
axs[0].set_ylim(0, 1.1)

sns.lineplot(ps, x="year", y="theta", ax=axs[1], color=colors[1])

axs[1].set_xlabel("Year")
axs[1].set_ylabel("$\\widehat \\theta$")

axs[0].text(
    -0.1,
    1.1,
    "A",
    transform=axs[0].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="center",
)

axs[1].text(
    -0.1,
    1.1,
    "B",
    transform=axs[1].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="center",
)

fig.savefig("figures/figure_2.pdf", bbox_inches="tight")
fig.savefig("figures/figure_2.png", dpi=600, bbox_inches="tight")


# make figure 3


def get_windows(size):
    x = np.arange(432, 2025 - size + 1)
    return zip(x, x + size)


# size of the temporal window
size = 200

# try to use cached data or recompute from scratch
if Path("./data/figure_3.csv").exists():
    print("using existing dataframe for figure 3")
    ps_windowed = pd.read_csv("./data/figure_3.csv")
else:
    print("computing data for figure 3")

    # fix the seed for each temporal window
    random.seed(232)
    args = [(s, e, random.randint(0, 2**16)) for s, e in get_windows(size)]

    # run slatkin's test for each window
    def run(s, e, seed):
        # subset to popes at the beginning of the window
        d = popes.query("start <= @s")

        # get n0, k0, initial frequency vector for the window
        year = d["start"].iloc[-1]
        pope = d["pope"].iloc[-1]
        initial_configuration = " ".join([str(x) for x in d["name"].value_counts()])
        n0 = d.shape[0]
        k0 = len(d["name"].unique())

        # subset to popes at the end of the window
        d_end = popes.query("start < @e")
        # get n, k, final frequency window
        end_configuration = " ".join([str(x) for x in d_end["name"].value_counts()])
        n = d_end.shape[0]
        k = len(d_end["name"].unique())

        # run conditional Slatkin test on the window
        cmd = f"ewens-sampler -j --seed {seed} -s {int(SAMPLES)} --initial-configuration '{initial_configuration}' --test '{end_configuration}'"

        ans = json.loads(sbp.getoutput(cmd))

        # get popes inside the window
        d_win = popes.query("start >= @s and start < @e")
        k_win = len(d_win["name"].unique())
        n_win = d_win.shape[0]

        win_counts = d_win["name"].value_counts()
        win_freqs = (win_counts / win_counts.sum()).values
        win_entropy = -(win_freqs * np.log(win_freqs)).sum()

        result = {
            "year": year,
            "pope": pope,
            "n": n,
            "k": k,
            "window_start": s,
            "window_end": e,
            "window_size": size,
            "n0": n0,
            "k0": k0,
            "n_win": n_win,
            "k_win": k_win,
            "win_entropy": win_entropy,
            "theta": ans["theta"],
        }
        result.update(slatkin_tails(ans))
        return result

    # run over all windows in Parallel
    rows = list(
        tqdm(
            Parallel(n_jobs=20, return_as="generator")(delayed(run)(*x) for x in args),
            total=len(args),
        )
    )

    # aggregate and save to cache
    ps_windowed = pd.DataFrame.from_records(rows)
    ps_windowed.to_csv("./data/figure_3.csv")

# plot
fig, axs = plt.subplots(
    1,
    3,
    figsize=(12, 3),
    constrained_layout=True,
    sharex=True,
    gridspec_kw={"wspace": 0.1},
)
colors = sns.xkcd_palette(["slate blue", "teal green", "rust red"])

dplot = ps_windowed.query("window_size == @size")

axs[0].fill_between(
    dplot["window_end"],
    1 - dplot["p_upper"],
    dplot["p_lower"],
    color=colors[0],
    alpha=0.5,
    linewidth=0.8,
    edgecolor=colors[0],
)
axs[0].plot(dplot["window_end"], dplot["p_lower"], color=colors[0])
axs[0].plot(dplot["window_end"], 1 - dplot["p_upper"], color=colors[0])

axs[0].set_xlabel("Year")
axs[0].set_ylabel("$P_E$, Slatkin exact test tail probability", fontsize="small")

axs[0].axhline(1, color="gray", linestyle="--")
axs[0].axhline(0.025, color="red", alpha=0.6)
axs[0].axhline(0.975, color="red", alpha=0.6)
axs[0].set_ylim(0, 1.1)

sns.lineplot(dplot, x="window_end", y="theta", ax=axs[1], color=colors[1])

axs[1].set_xlabel("Year")
axs[1].set_ylabel("$\\widehat \\theta$")


sns.lineplot(dplot, x="window_end", y="win_entropy", ax=axs[2], color=colors[2])
axs[2].set_xlabel("Year")
axs[2].set_ylabel("Entropy of names in window")

axs[0].text(
    -0.1,
    1.1,
    "A",
    transform=axs[0].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="center",
)

axs[1].text(
    -0.1,
    1.1,
    "B",
    transform=axs[1].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="center",
)

axs[2].text(
    -0.1,
    1.1,
    "C",
    transform=axs[2].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="center",
)

fig.savefig("figures/figure_3.png", dpi=600, bbox_inches="tight")
fig.savefig("figures/figure_3.pdf", bbox_inches="tight")
