import numpy as np
import pandas as pd
import scipy.io as sio
import os
import json
import polars as pl


def list_matlab_files(source_dir, batch_name):
    """
    Lists matlab files in the folder
    """
    return sorted(
        [f for f in os.listdir(f"{source_dir}/{batch_name}") if f.endswith(".mat")]
    )


def read_matlab_file(mat_path):
    mat_data = sio.loadmat(mat_path, spmatrix=False)  # get file contents
    mat_headers = list(mat_data.keys())  # get file headers
    metadata = {}
    data = {}
    for header in mat_headers:
        if not header.startswith("__"):
            d = mat_data[header]
            if d.size == 1:
                value = d[0, 0]
                if np.issubdtype(value, np.integer):
                    clean_value = int(value)
                elif np.issubdtype(value, np.floating):
                    clean_value = float(value)
                metadata[header] = clean_value
            else:
                data[header] = d.flatten()
    data = pl.from_dict(data)
    return metadata, data


if __name__ == "__main__":
    source_dir = "matlab_data"
    output_dir = "outputs"
    batch_name = "2026_10_9_11h42"
    matlab_file_names = list_matlab_files(source_dir, batch_name)

    for mat_fname in matlab_file_names: 
        mat_root = mat_fname[:-4]
        mat_path = f"{source_dir}/{batch_name}/{mat_fname}"
        metadata, data = read_matlab_file(mat_path=mat_path)
        output_data_path = f"{output_dir}/{batch_name}/{mat_root}/{mat_root}.parquet"
        os.makedirs(f"{output_dir}/{batch_name}/{mat_root}", exist_ok=True)
        data.write_parquet(output_data_path)
        output_metadata_path = f"{output_dir}/{batch_name}/{mat_root}/{mat_root}.json"
        with open(output_metadata_path, "w") as mf:
            json.dump(metadata, mf)
    
