import numpy as np
import pandas as pd
import scipy.io as sio
import os
import json
import polars as pl
from math import pi


def calculate_rms(signal_name): 
    s = data.get_column(signal_name).to_numpy()
    sm = s.mean()
    srms = np.sqrt(((s -  sm)**2).mean())
    return srms

def phase_shift(d):
    time = d["time_meas"]
    acc1 = d["acc1_meas_vect"]
    acc2 = d["acc2_meas_vect"]
    bacc1 = np.trapezoid((acc1-np.mean(np.asarray(acc1)))*np.sin(2*pi*f*time), x=time)
    aacc1 = np.trapezoid((acc1-np.mean(np.asarray(acc1)))*np.cos(2*pi*f*time), x=time)
    bacc2 = np.trapezoid((acc2-np.mean(np.asarray(acc2)))*np.sin(2*pi*f*time), x=time)
    aacc2 = np.trapezoid((acc2-np.mean(np.asarray(acc2)))*np.cos(2*pi*f*time), x=time)
    phase_shift = np.atan(bacc1/aacc1)-np.atan(bacc2/aacc2)
    return phase_shift

# if __name__ == "__main__":
input_dir = "outputs"
batch_name = "2026_9_28_15h08"
batch_dir = f"{input_dir}/{batch_name}"
tests = sorted([d for d in os.listdir(f"{batch_dir}") if os.path.isdir(f"{batch_dir}/{d}")])
outputs = {k:[] for k in ["test", "time_meas", "Rload", "f", "pow1", "pow2", "v1rms", "v2rms", "acc1rms", "acc2rms", "xrms", "dotxrms", "acc1_temp", "acc2_temp", "phase_shift"]}

for test in tests:
    data = pl.read_parquet(f"{batch_dir}/{test}/{test}.parquet")
    with open(f"{batch_dir}/{test}/{test}.json") as ml:
        metadata = json.load(ml)
    Rload = float(metadata["Rload"])
    f     = float(metadata['f'])

    v1rms = calculate_rms("tension1_meas_vect")
    v2rms = calculate_rms("tension2_meas_vect")
    pow1 = v1rms**2 / Rload
    pow2 = v2rms**2 / Rload
    outputs["test"].append(test) 
    outputs["time_meas"].append(data.get_column("time_meas").to_numpy())
    outputs["f"].append(f)
    outputs["Rload"].append(Rload) 
    outputs["pow1"].append(pow1)
    outputs["pow2"].append(pow2)
    outputs["v1rms"].append(v1rms)
    outputs["v2rms"].append(v2rms)
    outputs["acc1rms"].append(calculate_rms('acc1_meas_vect'))
    outputs["acc2rms"].append(calculate_rms('acc2_meas_vect'))
    outputs["xrms"].append(calculate_rms("dep_meas_vect"))
    outputs["dotxrms"].append(calculate_rms("vit_meas_vect"))
    outputs["acc1_temp"].append(np.asarray(data.get_column("acc1_meas_vect").to_numpy()))
    outputs["acc2_temp"].append(np.asarray(data.get_column("acc2_meas_vect").to_numpy()))
    outputs["phase_shift"].append(phase_shift(data))

data = pl.from_dict(outputs)