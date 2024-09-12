from pathlib import Path
from random import shuffle
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

# path to the nsathermocldphaseC1.c1 files
INPUT_DATA_PATH = Path("./data/raw/nsathermocldphaseC1.c1")

# path to where the monthly nsathermocldphaseC1.c1 files will be saved. each nc file has
# dimensions of (samples, time_idx, height_idx).
MONTHLY_NC_OUTPUTS = Path("./data/cnn_inputs/monthly_chunks")
MONTHLY_NC_OUTPUTS.mkdir(exist_ok=True, parents=True)


# the variables we want to preserve from the raw thermocldphase.c1 data.
FEATURE_VARS = [
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
CNN_VARS = ["cloud_phase_mplgr"] + FEATURE_VARS


def get_monthly_file_groups(files: list[Path]) -> list[list[Path]]:
    """Collects files into 1-month groups. Relies on ARM file naming conventions."""

    def get_yyyymm_str(fp: Path) -> str:
        return fp.name.split(".")[2][:6]

    if len(files) == 1 and files[0].is_dir():
        files = list(files[0].glob("*.nc"))
    files = sorted(files)
    file_groups = [[files[0]]]
    fg_i = 0
    for i, file in enumerate(files[1:]):
        if get_yyyymm_str(file) == get_yyyymm_str(files[i]):
            file_groups[fg_i].append(file)
        else:
            file_groups.append([file])
            fg_i += 1
    return file_groups


def is_valid_chunk(chunk: xr.Dataset, n_time: int):
    has_enough_time = len(chunk["time"]) >= n_time
    has_enough_clouds = (chunk["cloud_phase_mplgr"] > 0).sum() > 100
    no_unknowns = (chunk["cloud_phase_mplgr"] != 8).all()
    return has_enough_time and has_enough_clouds and no_unknowns


def extract_samples(
    dataset: xr.Dataset, chunk_size: tuple[int, int] | None = None
) -> xr.Dataset:
    """Returns a dataset with a primary coordinate of sample (start datetime)"""
    if chunk_size is None:
        chunk_size = (128, 384)  # time X height

    n_time, n_height = chunk_size

    time_min = dataset["time"].min().values
    time_max = dataset["time"].max().values
    full_time_range = pd.date_range(start=time_min, end=time_max, freq="30s")  # type: ignore
    ds = dataset.reindex({"time": full_time_range}, fill_value=np.nan)

    chunks: list[xr.Dataset] = []
    for i in tqdm(range(len(ds["time"]) // n_time)):
        chunk = ds.isel(
            time=slice(i * n_time, (i + 1) * n_time),
            height=slice(0, n_height),
        )
        if not is_valid_chunk(chunk, n_time=n_time):
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
    out = xr.concat(chunks, dim="sample")
    for var_name in FEATURE_VARS:
        out[var_name] = (
            out[var_name].fillna(-9999).astype(np.float32)
        )  # float16 not supported

    out["cloud_flag"] = out["cloud_flag"].fillna(-1).astype(np.int8)
    out["cloud_phase_mplgr"] = out["cloud_phase_mplgr"].fillna(0).astype(np.int8)

    return out


def preprocess(daily_ds: xr.Dataset) -> xr.Dataset:
    """Limits the data to just the desired variables and broadcasts lwp to (time, height)"""
    daily_ds = daily_ds[CNN_VARS]
    daily_ds["mwrret1liljclou_be_lwp"] = daily_ds[
        "mwrret1liljclou_be_lwp"
    ].broadcast_like(daily_ds["temp"])
    return daily_ds


def save_dataset(ds: xr.Dataset, folder: Path):
    float_vars = [v for v in ds.data_vars if "float" in ds[v].dtype.str]
    encoding = {  # save some disk space. normally 1-2 GB per file
        v: {"dtype": "float32", "zlib": True, "complevel": 4} for v in float_vars
    }
    yyyy_mm = ds["sample"][0].dt.strftime("%Y_%m").item()
    output_filepath = folder / f"{folder.name}_{yyyy_mm}.20240221.nc"
    ds.to_netcdf(output_filepath, encoding=encoding)
    print(f"wrote {output_filepath.as_posix()}")
    return output_filepath


def write_monthly_chunked_files(files: list[Path]) -> list[Path]:
    outputs: list[Path] = []  # one file per month
    file_groups = get_monthly_file_groups(files)
    for file_group in tqdm(file_groups):
        ds = xr.open_mfdataset(file_group, preprocess=preprocess)
        samples = extract_samples(ds, chunk_size=(128, 384)).load()
        output_filepath = save_dataset(samples, MONTHLY_NC_OUTPUTS)
        outputs.append(output_filepath)
        samples.close()
        ds.close()
        del samples
        del ds
    return outputs


def get_train_valid_test_files(
    monthly_files: list[Path],
) -> tuple[list[Path], list[Path], list[Path]]:
    """splits the chunked monthly files into training, validation, and test sets. 2021
    data is hardcoded to be reserved for testing. the remainder of the monthly files
    (2018-2020) are shuffled and split 80/20 into training and validation sets."""
    train_valid_files, test_files = [], []
    for file in monthly_files:
        if ".2021" in file.name:
            test_files.append(file)
        elif (".2018" in file.name) or (".2019" in file.name) or (".2020" in file.name):
            train_valid_files.append(file)
    shuffle(train_valid_files)  # in-place
    train_files = train_valid_files[: int(0.8 * len(train_valid_files))]
    valid_files = train_valid_files[int(0.8 * len(train_valid_files)) :]
    return train_files, valid_files, test_files


def to_cnn_files(
    files: list[Path], label: Literal["train", "valid", "test"]
) -> tuple[Path, Path]:
    """takes a list of files for the training, validation, or test set and returns the
    paths to the X and y .npy files that can be directly used by the cnn models.

    X has shape (samples, time, height, len(features))
    y has shape (samples, time, height, 1), with values 0-7
    """
    ds = xr.open_mfdataset(files)

    normalizations = {
        "cloud_flag": lambda x: np.clip(x, 0, 1),  # 0=clear, 1=cloud
        "temp": lambda x: (np.clip(x, -100, 50) + 30) / 30,
        "mpl_backscatter": lambda x: (np.log(np.clip(x, 1e-8, 1e4)) + 6) / 8,
        "mpl_linear_depol_ratio": lambda x: np.clip(x, 0, 1) * 2 - 1,
        "reflectivity": lambda x: (np.clip(x, -70, 70) + 20) / 30,
        "radar_linear_depolarization_ratio": lambda x: np.clip(x + 20, -20, 20) / 6,
        "spectral_width": lambda x: np.clip(x * 5, -1, 4) - 0.5,
        "mean_doppler_velocity": lambda x: np.clip(x + 0.5, -8, 8) / 2,
        "mwrret1liljclou_be_lwp": lambda x: (np.log(np.clip(x, 0.1, 2000)) - 3) / 2,
    }

    y = np.expand_dims(ds["cloud_phase_mplgr"].values, axis=-1)
    X = np.stack(
        [
            normalizations[v](ds[v].fillna(-9999).values)  # -9999 will be clipped
            for v in FEATURE_VARS
        ],
        axis=-1,
    )

    x_path = Path(f"./data/cnn_inputs/X_{label}.npy")
    y_path = Path(f"./data/cnn_inputs/y_{label}.npy")
    np.save(x_path, X)
    np.save(y_path, y)

    return x_path, y_path


if __name__ == "__main__":
    files = list(INPUT_DATA_PATH.glob("*.nc"))
    assert len(
        files
    ), f"Please put the raw files directly into {INPUT_DATA_PATH.resolve()}"
    monthly_files = write_monthly_chunked_files(files)
    train_files, valid_files, test_files = get_train_valid_test_files(monthly_files)

    X_tr_p, y_tr_p = to_cnn_files(train_files, "train")
    print(f"Wrote training dataset to {X_tr_p}, {y_tr_p}.")

    X_v_p, y_v_p = to_cnn_files(valid_files, "valid")
    print(f"Wrote validation dataset to {X_v_p}, {y_v_p}.")

    X_te_p, y_te_p = to_cnn_files(test_files, "test")
    print(f"Wrote testing dataset to {X_te_p}, {y_te_p}.")

    print("done!")
