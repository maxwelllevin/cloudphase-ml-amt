# Preprocesses the netCDF data in `data/raw/nsathermocldphaseC1.c1` for RF and MLP model
# training. The output of this script is a collection of monthly parquet files where
# each file contains
#
# Usage:
# python netcdf_to_parquet.py
#

import functools
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from sklearn.utils import shuffle  # type: ignore

NaN = float("nan")

# This is where the data used to train and validate the RF and MLP models are kept. Note
# the test data is not
TRAIN_CACHE = Path("")
VALID_CACHE = Path()

# path to the nsathermocldphaseC1.c1 files
INPUT_DATA_PATH = Path("./data/raw/nsathermocldphaseC1.c1")

# This is where we'll keep monthly parquet files consisting of just the variables used
# to train the RF and MLP models. These files aren't used directly by the models; they
# are just kept for intermediate use for performance reasons. We found that loading the
# data all at once in xarray (from the netcdf files) quickly runs into memory/speed
# issues, whereas reading it in chunks and writing to parquet is much faster (though it
# still isn't fast).
MONTHLY_PARQUET_PATH = Path("./data/preprocessing/monthly")


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


def exception_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            error_traceback = traceback.format_exc()
            return error_traceback

    return wrapper


def parallel_process(
    function: Any, file_groups: list[list[Path]], workers: int
) -> list[Any]:
    """Applies a function in parallel on each file group."""
    outputs: list[Any] = []
    func = exception_logger(function)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(func, group) for group in file_groups}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                outputs.append(result)
            except Exception:
                # Get the traceback information as a string
                error_traceback = traceback.format_exc()
                print("Error occurred while processing:")
                print(error_traceback)
    return outputs


def preprocess_cloud_phase_month(files_for_month: list[Path]) -> Path:
    ds = xr.open_mfdataset(files_for_month).load()
    idxs = np.where(
        (ds.cloud_phase_mplgr.data > 0)
        & (np.isin(ds.qc_cloud_phase_mplgr.data, [0, 4]))
    )
    data = {
        "time": ds.time.dt.strftime("%Y-%m-%d %H:%M:%S").data[idxs[0]],
        "height": ds.height.data[idxs[1]],
        "cloud_phase_mplgr": ds.get("cloud_phase_mplgr", NaN),
        "qc_cloud_phase_mplgr": ds.get("qc_cloud_phase_mplgr", NaN),
        "temp": ds.get("temp", NaN),
        "mpl_backscatter": ds.get("mpl_backscatter", NaN),
        "mpl_linear_depol_ratio": ds.get("mpl_linear_depol_ratio", NaN),
        "reflectivity": ds.get("reflectivity", NaN),
        "radar_linear_depolarization_ratio": ds.get(
            "radar_linear_depolarization_ratio", NaN
        ),
        "spectral_width": ds.get("spectral_width", NaN),
        "mean_doppler_velocity": ds.get("mean_doppler_velocity", NaN),
        "mwrret1liljclou_be_lwp": ds.get("mwrret1liljclou_be_lwp", NaN),
    }
    df = pd.DataFrame(data)
    df = df.set_index(["time", "height"]).dropna(how="all", axis="columns")
    if "temp" not in df.columns:
        print("WARNING: No temperature data???")

    timestamp = ds.time[0].dt.strftime("%Y%m%d.%H%M%S").item()
    site = files_for_month[0].name[:3]
    output_path = MONTHLY_PARQUET_PATH / f"{site}.{timestamp}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)

    ds.close()
    del ds
    del df

    return output_path


def write_monthly_parquet_files(files: list[Path], workers: int = 4) -> list[Path]:
    """Aggregate a collection of files. Glob * patterns are allowed.

    Grabs the site name from the first three characters of the first file.

    Saves the outputs to ./data/agg/phase/{site}/*.parquet

    Output parquet files are indexed by time and height.

    Returns the paths to the output files.
    """
    file_groups = get_monthly_file_groups(files)
    output_files = parallel_process(
        preprocess_cloud_phase_month, file_groups, workers=workers
    )
    return sorted(output_files)


def write_train_valid_data(monthly_parquet_files: list[Path]) -> tuple[Path, Path]:
    N_TRAIN = 200_000  # per phase category
    N_VALID = 50_000  # per phase category

    phase_map = phase_map = {
        0: "clear",
        1: "liquid",
        2: "ice",
        3: "mixed",
        4: "drizzle",
        5: "liq_driz",
        6: "rain",
        7: "snow",
        8: "unknown",
    }

    data = pd.concat([pd.read_parquet(file) for file in monthly_parquet_files])
    data = data.reset_index()
    data = data[data["time"].between("2018-01-01", "2020-12-31")]
    data = data.dropna(subset=["cloud_phase_mplgr"])
    data["cloud_phase_mplgr"] = (
        data["cloud_phase_mplgr"].astype(int).map(phase_map).astype("category")
    )

    phase_data = {}
    for phase in tqdm.tqdm(list(phase_map.values())[1:]):  # none that are type 'clear'
        phase_data[phase] = data[data["cloud_phase_mplgr"] == phase]
        if len(phase_data[phase]) < max(N_TRAIN, N_VALID):
            print(
                f"WARNING: not enough {phase} points. Found {len(phase_data[phase])},"
                f" but wanted {max(N_TRAIN, N_VALID)}"
            )

    # Join into training/validation sets
    _train: dict[str, pd.DataFrame] = {}
    _valid: dict[str, pd.DataFrame] = {}

    for phase in tqdm.tqdm(phase_data):  # none that are type 'clear'
        sample = phase_data[phase].sample(
            min(N_TRAIN + N_VALID, len(phase_data[phase]))
        )
        _train[phase] = sample.iloc[:N_TRAIN]
        _valid[phase] = sample.iloc[N_TRAIN : N_TRAIN + N_VALID]

    train: pd.DataFrame = shuffle(pd.concat(_train.values())).reset_index(drop=True)  # type: ignore
    valid: pd.DataFrame = shuffle(pd.concat(_valid.values())).reset_index(drop=True)  # type: ignore

    # Save to disk
    TRAIN_CACHE.parent.mkdir(exist_ok=True)
    train.to_parquet(TRAIN_CACHE)
    valid.to_parquet(VALID_CACHE)

    return TRAIN_CACHE, VALID_CACHE


if __name__ == "__main__":
    files = list(INPUT_DATA_PATH.glob("*.nc"))
    monthly_parquet_files = write_monthly_parquet_files(files)
    train, valid = write_train_valid_data(monthly_parquet_files)
