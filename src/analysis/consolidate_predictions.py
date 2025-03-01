# Consolidates predictions matching glob pattern on the command line (defaults to 2021
# NSA test set) into two files:
# 1. data/{label}_cloudy_predictions.parquet -- predictions for both CNN models, RF, and
#       MLP dimensioned by time & height. Includes cloud_phase_mplgr labels. Excludes
#       pixels where it is not cloudy or where the real phase is unknown.
# 2. data/{label}_phase_counts.parquet -- counts of cloud phases by height for both CNN
#       models, RF, and MLP. Columns are "height", "variable", and "count"
#
# TO RUN THIS SCRIPT:
#
# python consolidate_predictions.py path/to/files/*glob*.nc

import concurrent.futures
from pathlib import Path

import pandas as pd
import xarray as xr
from tqdm import tqdm

PHASE_MAP = {
    1: "liquid",
    2: "ice",
    3: "mixed",
    4: "drizzle",
    5: "liq_driz",
    6: "rain",
    7: "snow",
}

MODELS = {
    "cnn_dropout": "cnn_dropout",
    "cnn": "cnn",
    "rf_balanced": "rf_balanced",
    "rf_imbalanced": "rf_imbalanced",
    "mlp_balanced": "mlp_balanced",
    "mlp_imbalanced": "mlp_imbalanced",
}
CONFIDENCE_VARS = {
    f"{old}_confidence": f"{new}_confidence" for old, new in MODELS.items()
}
ABLATION_VARS = {
    f"{old_name}_{suffix}": f"{new_name}_{suffix}"
    for old_name, new_name in MODELS.items()
    for suffix in [
        "mpl",
        "mpl_b",
        "mpl_ldr",
        "mwr",
        "rad",
        "rad_ldr",
        "rad_mdv",
        "rad_ref",
        "rad_spec",
        "sonde",
    ]
}
CLDPHASE_VARS = {
    "cloud_phase_mplgr": "cloud_phase",
    **MODELS,
    **CONFIDENCE_VARS,
    **ABLATION_VARS,
}

OUT_DIR = Path(__file__).parent / "data"


def process_file(file: Path) -> pd.DataFrame:
    _ds = xr.open_dataset(file)
    for var in list(CONFIDENCE_VARS):
        _ds[var] = _ds[var][:, :, 1:].max(axis=-1)
    _ds = _ds[list(CLDPHASE_VARS) + ["cloud_flag"]]
    _df = _ds.to_dataframe()
    _df = _df[_df["cloud_phase_mplgr"].isin([1, 2, 3, 4, 5, 6, 7])]
    _df = _df[_df["cloud_flag"] == 1.0]
    _df.pop("cloud_flag")
    _df = _df.rename(columns=CLDPHASE_VARS)
    for col in _df.columns:
        if str(col) in list(CONFIDENCE_VARS.values()):
            continue
        _df[col] = pd.Categorical(
            _df[col].map(PHASE_MAP),
            categories=PHASE_MAP.values(),
            ordered=True,
        )
    _ds.close()
    return _df


def process_files(files: list[Path], label: str) -> None:
    pred_path = OUT_DIR / f"parallel_{label}_cloudy_predictions.parquet"
    count_path = OUT_DIR / f"parallel_{label}_phase_counts.parquet"

    print("loading datasets...")
    predictions: list[pd.DataFrame] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        for _df in tqdm(executor.map(process_file, files), total=len(files)):
            predictions.append(_df)

    print(f"saving predictions on known cloud pixels to {pred_path}...")
    pred_df = pd.concat(predictions)
    pred_df.to_parquet(pred_path)

    print("working on phase counts by height...")
    COUNT_VARS = {"cloud_phase_mplgr": "cloud_phase", **MODELS}
    df = pred_df[list(COUNT_VARS.values())].reset_index().drop("time", axis=1)
    _melt = df.melt(id_vars=["height"], var_name="variable", value_name="phase")
    _result = _melt.groupby(["height", "variable", "phase"]).size()
    count_df = pd.DataFrame(dict(count=_result))
    print(f"saving phase counts by height to {count_path}...")
    count_df.to_parquet(count_path)

    print("done!")


if __name__ == "__main__":
    import sys

    default_fpath = Path(__file__).parent.parent / "processing/data/predictions/"
    default_files = sorted(default_fpath.glob("nsa*2021*.nc"))

    files = [Path(f) for f in sys.argv[1:]] if len(sys.argv) > 1 else default_files
    label = files[0].name[:3]

    process_files(files=files, label=label)
