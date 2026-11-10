#!/usr/bin/env python

import json
import subprocess as sbp
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm


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
        icstr = f"--initial-configuration '{initial_configuration}' "

    cmd = f"./ewens-sampler -q -j --seed {seed} {bstr}{icstr}-s {int(samples)} --test '{configuration}'"

    ans = json.loads(sbp.getoutput(cmd))

    s = next(x for x in ans["tests"] if x["test"] == "slatkin")
    return s["p_lower"], s["p_upper"]


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

    cmd = f"./ewens-sampler -q -n {n} -k {k} --seed {seed} {bstr}{icstr}-s {int(samples)}"

    s = sbp.getoutput(cmd).split("\n")
    return s


def run_power(d, samples=1_000, test_samples=10_000):
    year = d["start"].iloc[-1]
    pope = d["pope"].iloc[-1]

    n, k = d.shape[0], len(d["name"].unique())
    rows = []
    for bias in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.11, 1.25, 1.43, 1.67, 2]:
        for i, tc in enumerate(
            sample_configurations(
                n, k, bias=bias, samples=samples, seed=int(n + k + (bias + 9) * 1000)
            )
        ):
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


def run_power_windows(popes, s, size, samples=1_000, test_samples=10_000):
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
