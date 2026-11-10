#!/usr/bin/env python

import json
import subprocess as sbp
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm.auto import tqdm


# run a single (full or conditional) exact test
def run_test_conditional(
    n,
    k,
    configuration,
    initial_configuration=None,
    bias=1.0,
    samples=1e7,
    seed=234,
):
    # build the call string
    if bias == 1.0:
        bstr = ""
    else:
        bstr = f"--bias {bias} "

    if initial_configuration is None:
        icstr = ""
    else:
        icstr = f"--initial-configuration '{initial_configuration}' "

    cmd = f"ewens-sampler -q -j --seed {seed} {bstr}{icstr}-s {int(samples)} --test '{configuration}'"

    ans = json.loads(sbp.getoutput(cmd))

    # return tail probabilities
    s = next(x for x in ans["tests"] if x["test"] == "slatkin")
    return s["p_lower"], s["p_upper"]


# sample configurations under the biased chinese restaurant process
def sample_configurations(
    n, k, initial_configuration=None, bias=1.0, samples=100, seed=231
):
    if bias == 1.0:
        bstr = ""
    else:
        bstr = f"--bias {bias} "

    if initial_configuration is None:
        icstr = ""
    else:
        icstr = f"--initial-configuration '{initial_configuration}' "

    cmd = f"ewens-sampler -q -n {n} -k {k} --seed {seed} {bstr}{icstr}-s {int(samples)}"

    s = sbp.getoutput(cmd).split("\n")
    return s


# power calculation for the exact test
def run_power(d, samples=1_000, test_samples=10_000):
    year = d["start"].iloc[-1]
    pope = d["pope"].iloc[-1]

    # get observed values of n and k
    n, k = d.shape[0], len(d["name"].unique())

    # for each value of bias
    rows = []
    for bias in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.11, 1.25, 1.43, 1.67, 2]:
        # first sample frequency vectors under the alternative model
        for i, tc in enumerate(
            sample_configurations(
                n, k, bias=bias, samples=samples, seed=int(n + k + (bias + 9) * 1000)
            )
        ):
            # and test for the null model
            p_lower, p_upper = run_test_conditional(
                n,
                k,
                tc,
                samples=test_samples,
                seed=int(n + k + (bias + 10) * 1000),
            )

            rows.append(
                {
                    "year": year,
                    "pope": pope,
                    "n": n,
                    "k": k,
                    "bias": bias,
                    "rep": i,
                    "p_lower": p_lower,
                    "p_upper": p_upper,
                }
            )

    return pd.DataFrame.from_records(rows)


# same power calculation but for the conditional test in temporal windows
def run_power_windows(popes, s, size, samples=1_000, test_samples=10_000):
    e = s + size
    d = popes.query("start <= @s")

    # get observed n0, k0, initial frequency vector
    year = d["start"].iloc[-1]
    pope = d["pope"].iloc[-1]
    initial_configuration = " ".join([str(x) for x in d["name"].value_counts()])
    n0 = d.shape[0]
    k0 = len(d["name"].unique())

    # and a final n and k at window end
    d_end = popes.query("start < @e")
    end_configuration = " ".join([str(x) for x in d_end["name"].value_counts()])
    n = d_end.shape[0]
    k = len(d_end["name"].unique())

    # for each bias value, first sample a frequency vector
    # completing the initial one under the alternative model
    rows = []
    for bias in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.11, 1.25, 1.43, 1.67, 2]:
        for i, tc in enumerate(
            sample_configurations(
                n,
                k,
                initial_configuration,
                bias=bias,
                samples=samples,
                seed=int(n + k + bias * 1000 + 2),
            )
        ):
            # then test for the null
            p_lower, p_upper = run_test_conditional(
                n,
                k,
                tc,
                initial_configuration,
                samples=test_samples,
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
                    "rep": i,
                    "p_lower": p_lower,
                    "p_upper": p_upper,
                }
            )
    return pd.DataFrame.from_records(rows)


# given simulation results, add the two tail probabilities
# to get fraction of rejections
def rejections(df, keys, alpha=0.05):
    df = df.assign(
        reject_lower=df["p_lower"] < alpha / 2,
        reject_upper=df["p_upper"] < alpha / 2,
    )
    df["reject"] = df["reject_lower"] | df["reject_upper"]
    return (
        df.groupby(keys)
        .agg(
            total=("reject", "size"),
            twotail=("reject", "mean"),
            lower=("reject_lower", "mean"),
            upper=("reject_upper", "mean"),
        )
        .reset_index()
    )


# read data
popes = pd.read_csv("popes-list.csv")


## compute panel A

if Path("./data/figure_s1a.csv").exists():
    print("using existing dataframe for figure S1A")
    power_df = pd.read_csv("./data/figure_s1a.csv")
else:
    args = [popes.iloc[:i, :] for i in range(44, popes.shape[0] + 1)]

    results = list(
        tqdm(
            Parallel(n_jobs=-1, return_as="generator")(
                delayed(run_power)(y) for y in args
            ),
            total=len(args),
        )
    )
    power_df = pd.concat(results)
    power_df.to_csv("./data/figure_s1a.csv", index=False)

power_df = rejections(power_df, ["year", "pope", "n", "k", "bias"])
power_df["bias"] = power_df["bias"].astype(str)

## compute panel B

if Path("./data/figure_s1b.csv").exists():
    print("using existing dataframe for figure S1B")
    power_windf = pd.read_csv("./data/figure_s1b.csv")
else:
    args = []
    size = 200
    for s in popes.query("start >= 432 and start < 2025 - @size + 5")["start"]:
        args.append((popes, s, size))

    results = list(
        tqdm(
            Parallel(n_jobs=-1)(delayed(run_power_windows)(*y) for y in args),
            total=len(args),
        )
    )
    power_windf = pd.concat(results)
    power_windf.to_csv("./data/figure_s1b.csv", index=False)

power_windf = rejections(
    power_windf, ["year", "pope", "n", "k", "window_start", "window_end", "bias"]
)
power_windf["bias"] = power_windf["bias"].astype(str)

# now we have power_df and power_windf
#  time to make the plot

fig, axs = plt.subplots(
    1, 2, figsize=(12, 4), layout="compressed", sharex=False, sharey=True
)

ax = axs[0]

sns.lineplot(
    power_df,
    x="year",
    y="twotail",
    hue="bias",
    ax=ax,
    palette=sns.color_palette("icefire", 11),
    legend=False,
)

ax.set_xlabel("Year")
ax.set_ylabel("Power: P(reject null | alternative true)")
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

sns.lineplot(
    power_windf,
    x="window_end",
    y="twotail",
    hue="bias",
    ax=ax,
    palette=sns.color_palette("icefire", 11),
)

ax.set_xlabel("Year")
ax.set_ylabel("Power: P(reject null | alternative true)")
ax.axhline(1, color="gray", linestyle="--")
ax.axhline(0, color="gray", linestyle="--")

ax.legend(
    title=("$\\beta$"),
    loc="upper left",
    bbox_to_anchor=(1, 1),
)

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

fig.savefig("figures/figure_s1.pdf", bbox_inches="tight")
fig.savefig("figures/figure_s1.png", bbox_inches="tight", dpi=600)
