# This script calculates permutation feature importances for each model and phase class
# on the testing dataset (NSA 2021).
#
# The "permutation importance" of a feature is the decrease in model accuracy occurring
# as a result of shuffling that feature's values in the test dataset. There are three
# main steps to this: 1) calculate a baseline accuracy score, 2) shuffle the feature's
# values and calculate the model's accuracy again, 3) report importance for the feature
# as baseline accuracy minus shuffled accuracy. "Accuracy" here is actually calculated
# for each phase class as TP / (TP + FN), more commonly known as recall.
#
# We do this, but slice the data and results several ways:
# 1. Overall importance, as described above for each cloud phase class
# 2. Importance by height, using the same process as above applied to the dataset at
#   each height bin, for each phase class
#
# This script takes several hours to run, so the importance files produced by this
# script are included in the repo under src/analysis/data/importance/.
#
# TO RUN THIS SCRIPT:
# cd src/analysis
# python calculate_importances.py
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Union

# Do this above the tensorflow imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"  # hide stuff
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # should fix most memory issues
warnings.simplefilter("ignore")

import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tqdm.auto import tqdm

keras.backend.clear_session()

MODELS = Path(__file__).parent / "models"
INPUTS = Path(__file__).parent.parent / "preprocessing/data"
OUTPUTS = Path(__file__).parent / "data/importances"

PHASE_MAP = {
    0: "clear",
    1: "liquid",
    2: "ice",
    3: "mixed",
    4: "drizzle",
    5: "liq_driz",
    6: "rain",
    7: "snow",
}
PHASE_TO_NUM_MAP = {
    "clear": 0,
    "liquid": 1,
    "ice": 2,
    "mixed": 3,
    "drizzle": 4,
    "liq_driz": 5,
    "rain": 6,
    "snow": 7,
}
PHASE_NUMS = [1, 2, 3, 4, 5, 6, 7]  # ignore cloudy and unknown pixels for accuracy
PHASES = [PHASE_MAP[i] for i in PHASE_NUMS]

FEATURES = [
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

ModelType = Union[MLPClassifier, RandomForestClassifier, keras.Model]


def load_data() -> tuple[np.ndarray, np.ndarray]:
    X_test = np.load(INPUTS / "cnn_inputs/X_test.npy")
    y_test = np.squeeze(np.load(INPUTS / "cnn_inputs/y_test.npy"))
    return X_test, y_test


def _load_model(
    filepath: str | Path,
) -> ModelType:
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


def make_sklearn_pred(
    model: MLPClassifier | RandomForestClassifier, X: np.ndarray
) -> np.ndarray:
    """From the input tensor/array, makes predictions using the model and returns an
    array with shape (samples, 128, 384). X should have shape (samples, 128, 384, 9)"""
    data = X.reshape((-1, 384, 9)).reshape((-1, 9))
    df = pd.DataFrame(data, columns=FEATURES)
    output = pd.Series(0, index=df.index)

    cloudy_pixels = df[df["cloud_flag"] == 1]
    pred = pd.Series(model.predict(cloudy_pixels[model.feature_names_in_]))
    # outputs are labels (eg 'ice'). We want numbers so we can compare with y_test
    pred = pred.map(PHASE_TO_NUM_MAP)  # type: ignore
    output.iloc[cloudy_pixels.index] = pred
    # reshape to match y_test
    return output.values.reshape((-1, 128, 384))  # type: ignore


def _get_pred_func(model_label: str) -> Callable[[ModelType, np.ndarray], np.ndarray]:
    def cnn_p(model: keras.Model, X: np.ndarray) -> np.ndarray:
        proba = model.predict(X, verbose=0, batch_size=4)  # type: ignore
        return np.argmax(proba, axis=-1) + 1

    if "cnn" in model_label:
        return cnn_p
    return make_sklearn_pred


def phase_acc_scores(
    y_pred: np.ndarray, phase_idxs: dict[int, Any]
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for phase_id, idxs in phase_idxs.items():
        if not len(idxs[0]):
            scores[PHASE_MAP[phase_id]] = np.nan
            continue
        scores[PHASE_MAP[phase_id]] = (y_pred[idxs] == phase_id).mean()
    scores["avg"] = np.nanmean(list(scores.values()))  # type: ignore
    return scores


def phase_acc_height_scores(
    y_true: np.ndarray, y_pred: np.ndarray, phase_values: dict[int, Any]
) -> pd.DataFrame:
    # y_true/pred should have shape (samples, 128, 384)
    matches = y_true == y_pred
    scores = pd.DataFrame({"overall": matches.mean(axis=(0, 1))})
    for phase_id, of_phase_type in phase_values.items():
        numerator = (matches & of_phase_type).sum(axis=(0, 1))
        denominator = of_phase_type.sum(axis=(0, 1))
        acc_for_phase = np.full_like(numerator, fill_value=np.nan, dtype=float)
        np.divide(numerator, denominator, out=acc_for_phase, where=denominator > 100)
        scores[PHASE_MAP[phase_id]] = acc_for_phase
    scores["overall"] = matches.mean(axis=(0, 1))
    return scores


def importance_from_scores(
    permuted_scores: dict[str, Any], phase_baseline: dict[str, float]
) -> pd.DataFrame:
    phase_importance_df = pd.DataFrame.from_dict(permuted_scores, orient="columns")
    phase_importance_df = (
        phase_importance_df.reset_index(names="phase")
        .melt(id_vars="phase", var_name="feature", value_name="acc")
        .set_index("phase")
    )
    phase_importance_df["importance"] = (
        phase_importance_df.reset_index()
        .apply(lambda row: phase_baseline[row["phase"]] - row["acc"], axis=1)
        .set_axis(phase_importance_df.index)
    )
    return phase_importance_df


def importance_from_height_scores(
    permuted_height_scores: dict[str, Any], phase_height_baseline: pd.DataFrame
) -> pd.DataFrame:
    phase_height_dfs = {}
    for phase in list(PHASES) + ["overall"]:
        df = pd.DataFrame({k: v[phase] for k, v in permuted_height_scores.items()})
        df = -df.subtract(phase_height_baseline[phase], axis=0)
        df = df.reset_index(names=["height_idx"])
        df = df.melt(id_vars=["height_idx"], value_name="importance")
        df["phase"] = phase
        phase_height_dfs[phase] = df
    phase_height_importance_df = pd.concat(phase_height_dfs.values())
    phase_height_importance_df["height"] = 0.16 + (
        0.03 * phase_height_importance_df.pop("height_idx")
    )  # type: ignore
    return phase_height_importance_df


def calculate_importance(
    model: ModelType,
    pred_func: Callable[[ModelType, np.ndarray], np.ndarray],
    label: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    IMPORTANCE_PATH = OUTPUTS / f"{label}_importance.parquet"
    HEIGHT_IMPORTANCE_PATH = OUTPUTS / f"{label}_height_importance.parquet"

    # For convenience / performance
    y_flat = y_test.ravel()
    phase_idxs = {phase_id: np.where(y_flat == phase_id) for phase_id in PHASE_NUMS}
    phase_values = {phase_id: (y_test == phase_id) for phase_id in PHASE_NUMS}

    # Baseline predictions
    pred = pred_func(model, X_test)
    y_pred = pred.ravel()

    phase_baseline = phase_acc_scores(y_pred, phase_idxs=phase_idxs)
    phase_height_baseline = phase_acc_height_scores(
        y_test,
        pred,
        phase_values=phase_values,
    )

    permuted_scores = {}  # dict of feature_var:dict[phase:]
    permuted_height_scores = {}  # dict of feature_var:dict[phase:]

    assert len(FEATURES) == X_test.shape[-1]
    for i, feature_name in tqdm(enumerate(FEATURES[1:])):
        i = i + 1  # make i start at 1 since we are skipping cloud_flag

        # Permute the selected feature column and make predictions on the test set
        X_permuted = X_test.copy()
        np.random.shuffle(X_permuted[:, :, :, i])
        y_pred_permuted = pred_func(model, X_permuted)

        # Calculate metrics for the predictions on the permuted dataset
        permuted_scores[feature_name] = phase_acc_scores(
            y_pred_permuted.ravel(),
            phase_idxs=phase_idxs,
        )
        permuted_height_scores[feature_name] = phase_acc_height_scores(
            y_true=y_test,
            y_pred=y_pred_permuted,
            phase_values=phase_values,
        )

    # Consolidate and save permutation feature importances
    phase_importance_df = importance_from_scores(
        permuted_scores=permuted_scores,
        phase_baseline=phase_baseline,
    )
    phase_importance_df.to_parquet(IMPORTANCE_PATH)

    # Consolidate and save permutation feature importances by height
    phase_height_importance_df = importance_from_height_scores(
        permuted_height_scores=permuted_height_scores,
        phase_height_baseline=phase_height_baseline,
    )
    phase_height_importance_df.to_parquet(HEIGHT_IMPORTANCE_PATH)

    return phase_importance_df, phase_height_importance_df


def main():
    OUTPUTS.mkdir(exist_ok=True, parents=True)

    print("loading data....")
    X_test, y_test = load_data()

    model_labels = {
        "cnn": MODELS / "cnn.20240501.090456.h5",
        "rf_balanced": MODELS / "rf_balanced.joblib",
        "rf_imbalanced": MODELS / "rf_imbalanced.joblib",
        "mlp_balanced": MODELS / "mlp_balanced.joblib",
        "mlp_imbalanced": MODELS / "mlp_imbalanced.joblib",
        "cnn_dropout": MODELS / "cnn.20240429.213223.h5",
    }
    for model_label, model_path in model_labels.items():
        print(f"Working on {model_label}...")
        model = _load_model(model_path)
        pred_func = _get_pred_func(model_label)
        _ = calculate_importance(
            model=model,
            pred_func=pred_func,
            label=model_label,
            X_test=X_test,
            y_test=y_test,
        )
    print("Done!")


if __name__ == "__main__":
    with tf.device("CPU"):
        main()
