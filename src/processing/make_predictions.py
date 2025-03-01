import argparse
import os
import sys
import warnings
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


# INPUT_DIR = Path("/data/home/levin/data/datastream/anx/anxthermocldphaseM1.c1/")
# INPUT_GLOB = "*.nc"
INPUT_DIR = (
    Path(__file__).parent.parent / "preprocessing/data/raw/nsathermocldphaseC1.c1/"
)
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

    # Make predictions and get 'pred_as_class' with shape (time, height). Note that we
    # first get confidence scores (cnn.predict()) and drop the first index (clear-sky)
    # to ensure that the model cannot mistake a cloudy pixel for a clear sky pixel.
    cnn_pred = cnn.predict(cnn_inputs, verbose=0)  # type: ignore
    pred_as_class = np.argmax(cnn_pred[..., slice(1, 8)], axis=-1) + 1

    pred_xr = xr.full_like(ds["cloud_phase_mplgr"], -1)
    pred_xr.data[:] = pred_as_class[:]
    where_clear = np.where(ds["cloud_flag"].data == 0)
    pred_xr.data[where_clear] = 0  # fill in clear-sky where the VAP knows it is clear

    if return_confidences:
        pred_confidences = (
            xr.full_like(ds["cloud_phase_mplgr"], fill_value=-1, dtype=float)
            .expand_dims({"phase": 8}, axis=-1)
            .copy()
        )  # copy to make it writable
        pred_confidences.data[:] = cnn_pred[:]
        return pred_xr, pred_confidences

    return pred_xr


def sklearn_model_pred(
    model: MLPClassifier | RandomForestClassifier,
    df: pd.DataFrame,
    reshape_like: xr.DataArray,
    missing: list[str] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    if missing is not None and len(missing):
        df = df.copy()
        df[missing] = 0
    _proba = model.predict_proba(df)
    proba_df = pd.DataFrame(
        _proba,
        index=df.index,
        columns=[PHASE_MAP[c] for c in model.classes_],  # type: ignore
    )[[1, 2, 3, 4, 5, 6, 7]]  # ensures dataframe has correct column ordering

    proba = proba_df.values
    pred = proba.argmax(axis=-1) + 1  # add 1 to get cloud phase classes (1-7)

    # Coerce pred into an xarray DataArray with the right shape
    pred_xr = xr.full_like(reshape_like, fill_value=-1, dtype=float).copy()
    pred_xr.data[:] = pred.reshape(pred_xr.shape)[:]
    where_clear_sky = np.where(reshape_like.values == 0)
    pred_xr.data[where_clear_sky] = 0

    # Coerce pred probabilities into xarray DataArray with the right shape
    pred_proba_xr = (
        xr.full_like(reshape_like, fill_value=-1, dtype=float)
        .expand_dims({"phase": 8}, axis=-1)
        .copy()
    )
    zeros = np.expand_dims(np.zeros(proba.shape[:-1]), axis=-1)
    proba_reshaped = np.hstack([zeros, proba]).reshape(pred_proba_xr.shape)
    pred_proba_xr.data[:] = proba_reshaped[:]

    return pred_xr, pred_proba_xr


def _load_model(filepath: str | Path) -> Any:
    filepath = Path(filepath)
    name = filepath.name
    if name.startswith("cnn"):
        print(f"loading cnn model: {name}...")
        return keras.models.load_model(filepath, compile=False)  # type: ignore
    elif name.endswith("joblib"):
        print(f"loading model: {name}...")
        return joblib.load(filepath)
    else:
        raise ValueError(f"unknown model: {name}")


def tensor_to_sklearn_inputs(tensor: tf.Tensor) -> pd.DataFrame:
    X = np.array(tensor).reshape(-1, tensor.shape[-1])  # type: ignore
    X_df = pd.DataFrame(
        data=X,
        columns=[
            "cloud_flag",
            "temp",
            "mpl_backscatter",
            "mpl_linear_depol_ratio",
            "reflectivity",
            "radar_linear_depolarization_ratio",
            "spectral_width",
            "mean_doppler_velocity",
            "mwrret1liljclou_be_lwp",
        ],
    )
    return X_df


def main():
    print("starting ML prediction script...")
    print(f"INPUTS: {INPUT_DIR}")
    print(f"OUTPUTS: {OUT_DIR}")

    files = sorted(INPUT_DIR.glob(INPUT_GLOB))
    print(f"found {len(files)} files.")

    if len(files) and (OUT_DIR / files[0].name).exists():
        print("WARNING: some data have already been processed.")
        if "y" != input("Continue processing? (some data will be not be rerun) [y/N]"):
            print("Exited without processing data.")
            return

    print("loading models...")
    model_labels = {
        "cnn_dropout": MODELS / "cnn.20240429.213223.h5",
        "cnn": MODELS / "cnn.20240501.090456.h5",
        "rf_balanced": MODELS / "rf_balanced.joblib",
        "rf_imbalanced": MODELS / "rf_imbalanced.joblib",
        "mlp_balanced": MODELS / "mlp_balanced.joblib",
        "mlp_imbalanced": MODELS / "mlp_imbalanced.joblib",
    }
    models = {label: _load_model(filepath) for label, filepath in model_labels.items()}

    print("making predictions...")

    for i, filepath in enumerate(track(files, update_period=60)):
        output_filepath = OUT_DIR / filepath.name
        if output_filepath.exists():
            continue

        print(f"working on {filepath.name}...")

        ds = load_data(filepath).isel(height=slice(0, 384))

        # Inputs for CNN models
        next_file = files[i + 1] if i + 1 < len(files) else None
        inputs, pad, (tmin, tmax) = prep_cnn_inputs(ds, next_file)
        X_df = tensor_to_sklearn_inputs(inputs)

        # Make predictions & get confidences
        for label, model in models.items():
            if label.startswith("cnn"):
                pred, conf = cnn_predict(model, inputs, pad, return_confidences=True)
            elif label.startswith("rf") or label.startswith("mlp"):
                pred, conf = sklearn_model_pred(
                    model, X_df, reshape_like=pad["cloud_phase_mplgr"]
                )
            else:
                raise ValueError(f"unrecognized model: {label}")
            pad[label] = pred
            pad[label + "_confidence"] = conf

        # Make predictions for instrument ablation study
        for ab, missing in ABLATION_STRATEGY.items():
            for label, model in models.items():
                if label.startswith("cnn"):
                    pred = cnn_predict(model, inputs, pad, missing=missing)
                elif label.startswith("rf") or label.startswith("mlp"):
                    pred, _ = sklearn_model_pred(
                        model,
                        X_df,
                        reshape_like=pad["cloud_phase_mplgr"],
                        missing=missing,
                    )
                else:
                    raise ValueError(f"unrecognized model: {label}")
                pad[label + ab] = pred

        ds = pad.sel(time=slice(tmin, tmax))
        ds.to_netcdf(output_filepath)


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
