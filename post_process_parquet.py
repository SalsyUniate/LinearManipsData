import numpy as np
import pandas as pd
import scipy.io as sio
import os
import json
import polars as pl


def calculate_rms(signal_name): 
    s = data.get_column(signal_name).to_numpy()
    sm = s.mean()
    srms = np.sqrt(((s -  sm)**2).mean())
    return srms


# if __name__ == "__main__":
input_dir = "outputs"
batch_name = "2026_9_28_15h08"
batch_dir = f"{input_dir}/{batch_name}"
tests = sorted([d for d in os.listdir(f"{batch_dir}") if os.path.isdir(f"{batch_dir}/{d}")])
outputs = {k:[] for k in ["test", "Rload", "f", "pow1", "pow2", "v1rms", "v2rms", "acc1rms", "acc2rms", "xrms", "dotxrms"]}

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
    outputs["f"].append(f)
    outputs["Rload"].append(Rload) 
    outputs["pow1"].append(pow1)
    outputs["pow2"].append(pow2)
    outputs["v1rms"].append(v1rms)
    outputs["v2rms"].append(v2rms)
    outputs["acc1rms"].append("acc1_meas_vect")
    outputs["acc2rms"].append("acc2_meas_vect")
    outputs["xrms"].append("dep_meas_vect")
    outputs["dotxrms"].append("vit_meas_vect")

data = pl.from_dict(outputs)
print('hello')