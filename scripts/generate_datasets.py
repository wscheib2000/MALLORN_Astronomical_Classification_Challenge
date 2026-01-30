import pathlib
import argparse
import pandas as pd
from reservoirpy.nodes import Reservoir
import numpy as np


def main(args):

    reservoir = Reservoir(
        units=500,
        lr=0.1,                 # leak rate (depends on dt!)
        sr=0.9,                 # spectral radius
        input_scaling=0.5,
        seed=42
    )

    train_paths = list(pathlib.Path(args.in_folder).glob("*/train_full_lightcurves.csv"))
    X_train, y_train = build_dataset(train_paths, reservoir)

    test_paths = list(pathlib.Path(args.in_folder).glob("*/test_full_lightcurves.csv"))
    X_test, y_test = build_dataset(test_paths, reservoir)
    
    # TODO: save X_train, y_train, X_test, y_test to files


def build_dataset(paths, reservoir):
    data = pd.read_csv(paths[0])
    columns = data['Filter'].unique()

    for path in paths:
        data = pd.read_csv(path)
        
        data["Time (MJD)"] = (data["Time (MJD)"]*2).round()/2  # Round to nearest 0.5

        object_ids = data['object_id'].unique()
        for obj_id in object_ids:
            sub_data = data.loc[data['object_id'] == obj_id]
            sub_data = pre_process_data(sub_data, columns)
            assert sub_data.shape[1] == 12

            # TODO: reset reservoir
            # TODO: run reservoir on subject sequence
            # TODO: aggregate states → z_i
            # TODO: store (z_i, label) in X_train, y_train
    # TODO: return X, y


def pre_process_data(data, col_order):
    dt = data.groupby(["object_id", "Filter"])["Time (MJD)"].diff().min()
    median_dt = data.groupby(["object_id", "Filter"])["Time (MJD)"].diff().median()
    max_gap = 2 * median_dt

    t_min = data["Time (MJD)"].min()
    t_max = data["Time (MJD)"].max()
    uniform_index = np.arange(t_min, t_max + dt, dt)
        
    wide = data.pivot(index='Time (MJD)', columns='Filter', values='Flux')
    wide = wide.reindex(columns=col_order)  # ensure column order
    
    mask = (~wide.isna()).astype(float)
    
    wide = wide.reindex(uniform_index)
    mask = mask.reindex(uniform_index, fill_value=0)
    
    # Linear interpolation only for small gaps
    # limit = number of consecutive NaNs to fill
    limit = int(max_gap / dt)
    wide_interpolated = wide.interpolate(
        method='linear',
        limit=limit,
        limit_direction='both'
    )
    
    # Optional: normalize per measurement channel
    mean = wide_interpolated.mean()
    std = wide_interpolated.std()
    wide_normalized = (wide_interpolated - mean) / (std + 1e-8)
    
    # Concatenate mask channels
    return np.concatenate(
        [wide_normalized.to_numpy(), mask.to_numpy()],
        axis=1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train reservoir computing model"
    )
    parser.add_argument("--in_folder", help="the input folder")
    parser.add_argument("--out_folder", help="the output folder")
    args = parser.parse_args()

    main(args)
