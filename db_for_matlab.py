import numpy as np
import pandas as pd
import scipy.io as sio
import os
import json
import polars as pl
from math import pi
from collections import namedtuple
import jax 
import jax.numpy as jnp


def wr1_argmax_over_R(P, n1):
    idx_P = np.argmax(P, axis=1)
    i_opt = np.unravel_index(idx_P, (n1)) # unravel indices
    return i_opt # return indices
 
def extract_indices_optr1(P_np):
    nW, n1 = P_np.shape
    i_opt = wr1_argmax_over_R(P_np, n1)
    R1_opt = r_list[i_opt]
    return R1_opt


data_uncoupled  = pl.read_parquet("data_complete/2026_10_21_11h44.parquet")

results_uncoupled = data_uncoupled.group_by(
    pl.col("f")
).agg(
    pl.col("pow1").max().alias("pow1max"),
    pl.col("pow2").max().alias("pow2max"),
    (pl.col('pow1')+pl.col('pow2')).max().alias('powtot')
)

Ropt_uncoupled_1 = []
Ropt_uncoupled_2 = []
for f in (results_uncoupled["f"]):
        p1 = results_uncoupled.filter(pl.col("f") == f)["pow1max"]
        p2 = results_uncoupled.filter(pl.col("f") == f)["pow2max"]
        d1 = data_uncoupled.filter((pl.col("f")==f) & (pl.col("pow1") == p1))["Rload"]
        d2 = data_uncoupled.filter((pl.col("f")==f) & (pl.col("pow2") == p2))["Rload"]
        Ropt_uncoupled_1.append(d1)
        Ropt_uncoupled_2.append(d2)


f_list = np.asarray(data_uncoupled["f"].unique())
r_list = np.asarray(data_uncoupled["Rload"].unique())

power1_exp = np.zeros((len(r_list),len(f_list)))
power2_exp = np.zeros((len(r_list),len(f_list)))

for i, f in enumerate(f_list):
    for j, r in enumerate(r_list): 
        d = data_uncoupled.filter((pl.col("f") == f), (pl.col("Rload") == r))
        power1_exp[j][i] = d["pow1"][0]
        power2_exp[j][i] = d["pow2"][0]


data = pl.DataFrame({"f":f_list, "r1": Ropt_uncoupled_1, "r2": Ropt_uncoupled_2, "p1":power1_exp, "p2":power2_exp})
data.write_parquet('db_for_ml/data.parquet')