import argparse
import os
import re
import sys
import warnings
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np  # type: ignore
import pandas as pd
import xarray as xr  # type: ignore
from rich.progress import track
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Do this above the tensorflow imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"  # hide stuff
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # should fix most memory issues


import keras
import tensorflow as tf

warnings.simplefilter("ignore")


# INPUT_DIR = Path("/data/home/levin/data/datastream/mos/mosthermocldphaseM1.c1/")
# INPUT_GLOB = "*.nc"
INPUT_DIR = Path(__file__).parent.parent / "preprocessing/data/raw"
INPUT_GLOB = "*c1.2021*.nc"

OUT_DIR = Path(__file__).parent / "data/predictions/"
OUT_DIR.mkdir(exist_ok=True, parents=True)

MODELS = Path(__file__).parent / "models"

TF_DEVICE = "CPU"  # cpu to avoid most memory constraints, but is slow

PHASE_MAP = {
    "clear": 0,
    "liquid": 1,
    "ice": 2,
    "mixed": 3,
    "drizzle": 4,
    "liq_driz": 5,
    "rain": 6,
    "snow": 7,
    "unknown": 8,
}


ABLATION_STRATEGY = {  # label: channels
    #
    "_mpl_b": ["mpl_backscatter"],
    "_mpl_ldr": ["mpl_linear_depol_ratio"],
    "_rad_ref": ["reflectivity"],
    "_rad_ldr": ["radar_linear_depolarization_ratio"],
    "_rad_spec": ["spectral_width"],
    "_rad_mdv": ["mean_doppler_velocity"],
    "_sonde": ["temp"],
    "_mwr": ["mwrret1liljclou_be_lwp"],
    "_mpl": ["mpl_backscatter", "mpl_linear_depol_ratio"],
    "_rad": [
        "reflectivity",
        "radar_linear_depolarization_ratio",
        "spectral_width",
        "mean_doppler_velocity",
    ],
}


CNN_FEATURES = [
    "cloud_flag",
    "temp",
    "mpl_backscatter",
    "mpl_linear_depol_ratio",
    "reflectivity",
    "radar_linear_depolarization_ratio",
    "spectral_width",
    "mean_doppler_velocity",
    "mwrret1liljclou_be_lwp",
]


def load_cnn(filepath: Path) -> keras.Model:
    return keras.models.load_model(filepath, compile=False)  # type: ignore


def load_rf_1600k() -> RandomForestClassifier:
    return joblib.load(MODELS / "rf_1600k.20240504.235919.joblib")


def load_mlp_1600k() -> MLPClassifier:
    return joblib.load(MODELS / "mlp_1600k_norm.20240505.051911.joblib")


def load_data(filepath: Path) -> xr.Dataset:
    return xr.open_dataset(filepath)


def get_chunks(ds: xr.Dataset, selected_variables: list[str]) -> xr.Dataset:
    """Re-chunks the dataset into samples for the CNN model.

    The input dataset must have dimensions of (time, height).

    The output dataset will be returned with shape (sample, time_idx, height_idx) and
    will only contain feature variables.
    """

    to_float32 = [
        v for v in selected_variables if v not in ["cloud_flag", "cloud_phase_mplgr"]
    ]

    # Make dimensions time X height instead of only time. Needed by model.
    ds["mwrret1liljclou_be_lwp"] = ds["mwrret1liljclou_be_lwp"].broadcast_like(
        ds["temp"]
    )

    # Reindex the dataset with padding
    chunk_size = (128, 384)
    n_time, n_height = chunk_size
    time_min = ds["time"].min().values
    periods = len(ds["time"]) + n_time  # add some padding
    full_time_range = pd.date_range(start=time_min, periods=periods, freq="30s")
    ds = ds.reindex({"time": full_time_range})

    # Chunk-creation loop.
    chunks = []
    for i in range(len(ds["time"]) // n_time):
        chunk = ds.isel(
            time=slice(i * n_time, (i + 1) * n_time), height=slice(0, n_height)
        )

        if len(chunk["time"]) < n_time:
            continue

        # Create time and height index dimensions for the chunk
        chunk["time_idx"] = xr.DataArray(np.arange(n_time), dims=["time"])
        chunk["height_idx"] = xr.DataArray(np.arange(n_height), dims=["height"])

        # Make the data be indexed by the new index values, but preserve the original
        # time and height coords as data values.
        chunk = chunk.swap_dims({"time": "time_idx", "height": "height_idx"})
        chunk["time_value"] = xr.DataArray(chunk["time"].values, dims=["time_idx"])
        chunk["height_value"] = xr.DataArray(
            chunk["height"].values, dims=["height_idx"]
        )
        chunk = chunk.drop_vars(["time", "height"])

        # Add the first time value as the sample dimension index
        chunk = chunk.expand_dims("sample").assign_coords(
            sample=[chunk["time_value"].values[0]]
        )
        chunks.append(chunk)
    ds: xr.Dataset = xr.concat(chunks, dim="sample")
    for var_name in to_float32:
        ds[var_name] = (
            ds[var_name].fillna(-9999).astype(np.float32)
        )  # float16 not supported

    ds["cloud_flag"] = ds["cloud_flag"].fillna(-1).astype(np.int8)
    ds["cloud_phase_mplgr"] = ds["cloud_phase_mplgr"].fillna(0).astype(np.int8)
    return ds


def get_input_tensor(ds: xr.Dataset) -> tf.Tensor:
    """Returns a tensor of features used to predict outputs

    The dataset provided must have dimensions of (sample, time_idx, height_idx).

    The returned tensor will have shape (sample, time_idx, height_idx, 9)."""

    ds["mwrret1liljclou_be_lwp"] = ds["mwrret1liljclou_be_lwp"].broadcast_like(
        ds["temp"]
    )
    # ds = get_chunks(ds, features)
    normalizations = {  # from Andrew -- thank you!
        "cloud_flag": lambda x: x,
        "temp": lambda x: (np.clip(x, -100, 50) + 30) / 30,
        "mpl_backscatter": lambda x: (np.log(np.clip(x, 1e-8, 1e4)) + 6) / 8,
        "mpl_linear_depol_ratio": lambda x: np.clip(x, 0, 1) * 2 - 1,
        "reflectivity": lambda x: (np.clip(x, -70, 70) + 20) / 30,
        "radar_linear_depolarization_ratio": lambda x: np.clip(x + 20, -20, 20) / 6,
        "spectral_width": lambda x: np.clip(x * 5, -1, 4) - 0.5,
        "mean_doppler_velocity": lambda x: np.clip(x + 0.5, -8, 8) / 2,
        "mwrret1liljclou_be_lwp": lambda x: (np.log(np.clip(x, 0.1, 2000)) - 3) / 2,
    }
    # X = np.stack([normalizations[v](ds[v].values) for v in features], axis=-1)
    data = [normalizations[v](ds[v].fillna(-9999).values) for v in CNN_FEATURES]
    X = np.expand_dims(data, axis=0)
    X = X.transpose(0, 2, 3, 1)  # reshape from (1, 9, time, h) to (1, time, h, 9)
    # assert X.shape == (1, 2994, 384, 9), f"{X.shape=}, not (1, 2994, 384, 9)"
    return tf.convert_to_tensor(X)


def rf_1600k_predict(
    rf: RandomForestClassifier,
    ds: xr.Dataset,
    missing: list[str] | None = None,
    return_confidences: bool = False,
) -> xr.DataArray:
    features = rf.feature_names_in_
    ds_sub = ds[["time", "height", "cloud_flag", *features]]
    df = ds_sub.to_dataframe().fillna(0)
    df = df[df["cloud_flag"] == 1]
    df.pop("cloud_flag")

    if missing:
        for var in missing:
            df[var] = 0

    out = xr.full_like(ds["cloud_phase_mplgr"], fill_value=0)
    pred_confidences = (
        xr.full_like(ds["cloud_phase_mplgr"], -1)
        .expand_dims({"phase": 8}, axis=-1)
        .copy()
    )

    if len(df) >= 1:
        inputs = df.reset_index(drop=True)  # drop time/height4
        pred = pd.Series(rf.predict(inputs), index=df.index).map(PHASE_MAP)
        index_grid = list(product(out.time.data, np.round(out.height.data, 2)))
        out.data[:] = pred.reindex(index_grid).fillna(0).to_xarray().data[:]

        if return_confidences:
            grid = list(
                product(out.time.data, np.round(out.height.data, 2), np.arange(8))
            )
            pred_proba = (
                pd.DataFrame(
                    rf.predict_proba(inputs),
                    index=df.index,
                    columns=[PHASE_MAP[c] for c in rf.classes_],
                )
                .melt(ignore_index=False)
                .reset_index()
                .set_index(["time", "height", "variable"])
            )
            pred_confidences.data[:] = (
                pred_proba["value"].reindex(grid).fillna(0).to_xarray().data[:]
            )

    if return_confidences:
        return out, pred_confidences

    return out


def mlp_1600k_predict(
    mlp: MLPClassifier,
    ds: xr.Dataset,
    missing: list[str] | None = None,
    return_confidences: bool = False,
) -> xr.DataArray:
    normalizations = {  # from Andrew -- thank you!
        "temp": lambda x: (np.clip(x, -100, 50) + 30) / 30,
        "mpl_backscatter": lambda x: (np.log(np.clip(x, 1e-8, 1e4)) + 6) / 8,
        "mpl_linear_depol_ratio": lambda x: np.clip(x, 0, 1) * 2 - 1,
        "reflectivity": lambda x: (np.clip(x, -70, 70) + 20) / 30,
        "radar_linear_depolarization_ratio": lambda x: np.clip(x + 20, -20, 20) / 6,
        "spectral_width": lambda x: np.clip(x * 5, -1, 4) - 0.5,
        "mean_doppler_velocity": lambda x: np.clip(x + 0.5, -8, 8) / 2,
        "mwrret1liljclou_be_lwp": lambda x: (np.log(np.clip(x, 0.1, 2000)) - 3) / 2,
    }
    if missing:
        for var in missing:
            normalizations[var] = lambda x: np.clip(x, 0, 0)

    features = list(normalizations.keys())
    ds_sub = ds[["time", "height", "cloud_flag", *features]]
    df = ds_sub.to_dataframe()  # .fillna(0)
    df = df[df["cloud_flag"] == 1]
    df.pop("cloud_flag")
    for var, norm in normalizations.items():
        df[var] = norm(df[var]).fillna(0)

    out = xr.full_like(ds["cloud_phase_mplgr"], fill_value=0)
    pred_confidences = (
        xr.full_like(ds["cloud_phase_mplgr"], -1)
        .expand_dims({"phase": 8}, axis=-1)
        .copy()
    )

    if len(df) >= 1:
        inputs = df.reset_index(drop=True)  # drop time/height
        pred = pd.Series(mlp.predict(inputs), index=df.index).map(PHASE_MAP)
        index_grid = list(product(out.time.data, np.round(out.height.data, 2)))
        out.data[:] = pred.reindex(index_grid).fillna(0).to_xarray().data[:]

        if return_confidences:
            grid = list(
                product(out.time.data, np.round(out.height.data, 2), np.arange(8))
            )
            pred_proba = (
                pd.DataFrame(
                    mlp.predict_proba(inputs),
                    index=df.index,
                    columns=[PHASE_MAP[c] for c in mlp.classes_],
                )
                .melt(ignore_index=False)
                .reset_index()
                .set_index(["time", "height", "variable"])
            )
            pred_confidences.data[:] = (
                pred_proba["value"].reindex(grid).fillna(0).to_xarray().data[:]
            )

    if return_confidences:
        return out, pred_confidences

    return out


def prep_cnn_inputs(
    ds: xr.Dataset, next_file: Path | None = None
) -> tuple[tf.Tensor, xr.Dataset, tuple[Any, Any]]:
    # Pad the dataset to be divisible by 128 (time) and at least 1 full day (2880)
    time_min, time_max = (
        ds["time"].min().values,
        ds["time"].max().values,
    )
    padded_time_range = pd.date_range(
        start=time_min,
        freq="30s",
        periods=2944,  # >2880 and divisible by 128
    )
    if next_file is not None:
        next_ds = xr.open_dataset(next_file)
        next_ds = next_ds.sel(time=slice(padded_time_range[0], padded_time_range[-1]))
        next_ds = next_ds.isel(height=slice(0, 384))
        ds = xr.concat([ds, next_ds], dim="time")
    padded = ds.reindex({"time": padded_time_range}, method="nearest", tolerance=1)

    # height_min = ds["height"].min().values
    # height_max = ds["height"].max().values
    # padded_height_range = np.array([0.03 * i + height_min for i in range(384)])
    # ds = ds.reindex({"time": padded_time_range, "height": padded_height_range})

    cnn_inputs = get_input_tensor(padded)

    return cnn_inputs, padded, (time_min, time_max)


def zero_out_channels(tensor: tf.Tensor, channels: list[int]) -> tf.Tensor:
    mask = tf.reduce_any(
        [tf.one_hot(ch, depth=tensor.shape[-1]) for ch in channels], axis=0
    )
    mask = tf.logical_not(mask)
    mask = tf.broadcast_to(mask, tf.shape(tensor))
    return tf.where(mask, tensor, tf.zeros_like(tensor))


def cnn_predict(
    cnn: keras.Model,
    cnn_inputs: tf.Tensor,
    ds: xr.Dataset,
    missing: list[str] | None = None,
    return_confidences: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    if missing:
        channels = [CNN_FEATURES.index(var) for var in missing]
        cnn_inputs = zero_out_channels(cnn_inputs, channels)

    cnn_pred = cnn.predict(cnn_inputs, verbose=0)  # type: ignore
    pred_as_class = np.argmax(cnn_pred, axis=-1)  # remove one-hot encoding

    pred_xr = xr.full_like(ds["cloud_phase_mplgr"], -1)
    pred_xr.data[:] = pred_as_class[:]

    if return_confidences:
        pred_confidences = (
            xr.full_like(ds["cloud_phase_mplgr"], fill_value=-1, dtype=float)
            .expand_dims({"phase": 8}, axis=-1)
            .copy()
        )  # copy to make it writable
        pred_confidences.data[:] = cnn_pred[:]
        return pred_xr, pred_confidences

    return pred_xr


def _load_model(label: str) -> Any:
    file_label = re.sub(r"_(\d{8})_(\d{6})", r".\1.\2", label)
    if label.startswith("cnn"):
        filepath = MODELS / f"{file_label}.h5"
        print(f"loading cnn model: {filepath.name}...")
        return load_cnn(filepath=filepath)
    elif "1600k" in label:
        filepath = MODELS / f"{file_label}.joblib"
        model_name = filepath.name.split("_")[0]  # e.g., 'rf' or 'mlp'
        print(f"loading {model_name} model: {filepath.name}...")
        return joblib.load(filepath)
    else:
        raise ValueError(f"unknown model: {label}")


def main():
    print("starting ML prediction script...")
    print(f"INPUTS: {INPUT_DIR}")
    print(f"OUTPUTS: {OUT_DIR}")

    print("loading models...")
    model_labels = [
        "cnn_20240429_213223",  # Best model that uses 2D spatial dropouts
        "cnn_20240501_090456",  # Best mode overall (no 2D spatial dropouts)
        "rf_1600k_20240514_033147",  # Best single-pixel RF model
        "mlp_1600k_20240514_052837",  # Best single-pixel MLP model
    ]
    models = {label: _load_model(label) for label in model_labels}

    print("making predictions...")
    files = sorted(INPUT_DIR.glob(INPUT_GLOB))

    for i, filepath in enumerate(track(files)):
        print(f"working on {filepath.name}...")

        ds = load_data(filepath).isel(height=slice(0, 384))

        # Inputs for CNN models
        next_file = files[i + 1] if i + 1 < len(files) else None
        inputs, pad, (tmin, tmax) = prep_cnn_inputs(ds, next_file)

        # Make predictions & get confidences
        for label, model in models.items():
            if label.startswith("cnn"):
                pred, conf = cnn_predict(model, inputs, pad, return_confidences=True)
            elif label.startswith("rf"):
                pred, conf = rf_1600k_predict(model, pad, return_confidences=True)
            elif label.startswith("mlp"):
                pred, conf = mlp_1600k_predict(model, pad, return_confidences=True)
            else:
                raise ValueError(f"unrecognized model: {label}")
            pad[label] = pred
            pad[label + "_confidence"] = conf

        # Make predictions for instrument ablation study
        for ab, missing in ABLATION_STRATEGY.items():
            for label, model in models.items():
                if label.startswith("cnn"):
                    pad[label + ab] = cnn_predict(model, inputs, pad, missing=missing)
                elif label.startswith("mlp"):
                    pad[label + ab] = mlp_1600k_predict(model, pad, missing=missing)
                elif label.startswith("rf"):
                    pad[label + ab] = rf_1600k_predict(model, pad, missing=missing)
                else:
                    raise ValueError(f"unrecognized model: {label}")

        ds = pad.sel(time=slice(tmin, tmax))
        ds.to_netcdf(OUT_DIR / filepath.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions on 2021 test data")
    parser.add_argument("--log", action="store_true", help="Log to a file")
    args = parser.parse_args()

    log_file = None
    if args.log:
        log_file = open("predict_logs.txt", "a", buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file

    with tf.device(TF_DEVICE):
        main()

    if log_file is not None:
        log_file.close()
