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

from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from keras import Model
from plot_utils import PHASE_MAP, PHASE_TO_NUM_MAP
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tqdm.notebook import tqdm

PHASE_NUMS = [1, 2, 3, 4, 5, 6, 7]  # ignore cloudy and unknown pixels for accuracy
PHASES = [PHASE_MAP[i] for i in PHASE_NUMS]


CNN_FEATURE_VARS = [  # The CNNs include 'cloud_flag' as a feature. single-pixel models do not.
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


def load_models() -> tuple[Model, Model, MLPClassifier, RandomForestClassifier]:
    def add_argmax(model: Model) -> Model:
        """Adds an argmax layer to the model to ensure outputs are 0-7 instead of an array
        with length 7. This makes it easier to compare with the y_test labels."""
        argmax_layer = keras.layers.Lambda(
            lambda x: tf.argmax(x, axis=-1), name="argmax_layer"
        )
        return Model(inputs=model.input, outputs=argmax_layer(model.output))

    cnn_dropout: Model = add_argmax(
        keras.models.load_model("../processing/models/cnn.20240429.213223.h5")
    )  # type: ignore
    cnn: Model = add_argmax(
        keras.models.load_model("../processing/models/cnn.20240501.090456.h5")
    )  # type: ignore
    mlp: MLPClassifier = joblib.load(
        "../processing/models/mlp_1600k.20240514.052837.joblib"
    )
    rf: RandomForestClassifier = joblib.load(
        "../processing/models/rf_1600k.20240514.033147.joblib"
    )
    return cnn_dropout, cnn, mlp, rf


def load_data() -> tuple[np.ndarray, np.ndarray]:
    X_test = np.load("../preprocessing/data/cnn_inputs/X_test.npy")
    y_test = np.squeeze(np.load("../preprocessing/data/cnn_inputs/y_test.npy"))
    return X_test, y_test


def make_rf_pred(rf: RandomForestClassifier, df: pd.DataFrame) -> pd.Series:
    return pd.Series(rf.predict(df), index=df.index, name="rf_pred").map(
        PHASE_TO_NUM_MAP
    )


def make_mlp_pred(mlp: MLPClassifier, X: np.ndarray) -> np.ndarray:
    """From the input tensor/array, makes predictions using the MLP model and returns
    an array with shape (samples, 128, 384). X should have shape (samples, 128, 384, 9)"""
    data = X.reshape((-1, 384, 9)).reshape((-1, 9))
    df = pd.DataFrame(data, columns=CNN_FEATURE_VARS)

    mlp_out = pd.Series(0, index=df.index)
    cloudy = df[df["cloud_flag"] == 1]
    pred = pd.Series(mlp.predict(cloudy[mlp.feature_names_in_]))
    pred = pred.map(
        PHASE_TO_NUM_MAP
    )  # output is labels (eg 'ice'). We want numbers so we can compare with y_test
    mlp_out.iloc[cloudy.index] = pred
    return mlp_out.values.reshape((-1, 128, 384))  # reshape to match y_test


def calculate_cnn_importances(
    cnn_dropout: Model, cnn: Model, X_test: np.ndarray, y_test: np.ndarray
):
    y_flat = y_test.ravel()

    models = {"cnn_dropout": cnn_dropout, "cnn": cnn}
    for model_label, model in models.items():
        PHASE_IMPORTANCE_PATH = Path(
            f"./data/importance/{model_label}_phase_importances.parquet"
        )
        PHASE_HEIGHT_IMPORTANCE_PATH = Path(
            f"./data/importance/{model_label}_phase_height_importance.parquet"
        )

        pred = model.predict(X_test, verbose=0)  # type: ignore
        y_pred = pred.ravel()

        PHASE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)

        phase_idxs = {phase_id: np.where(y_flat == phase_id) for phase_id in PHASE_NUMS}
        phase_values = {phase_id: (y_test == phase_id) for phase_id in PHASE_NUMS}

        def phase_acc_scores(y_pred):
            scores: dict[str, float] = {}
            for phase_id, idxs in phase_idxs.items():
                if not len(idxs[0]):
                    scores[PHASE_MAP[phase_id]] = np.nan
                    continue
                scores[PHASE_MAP[phase_id]] = (y_pred[idxs] == phase_id).mean()
            scores["avg"] = np.nanmean(list(scores.values()))  # type: ignore
            return scores

        def phase_acc_height_scores(y_true, y_pred):
            # y_true/pred should have shape (samples, 128, 384)
            matches = y_true == y_pred
            scores = pd.DataFrame({"overall": matches.mean(axis=(0, 1))})
            for phase_id, of_phase_type in phase_values.items():
                numerator = (matches & of_phase_type).sum(axis=(0, 1))
                denominator = of_phase_type.sum(axis=(0, 1))
                acc_for_phase = np.full_like(numerator, fill_value=np.nan, dtype=float)
                np.divide(
                    numerator, denominator, out=acc_for_phase, where=denominator > 100
                )
                scores[PHASE_MAP[phase_id]] = acc_for_phase
            scores["overall"] = matches.mean(axis=(0, 1))
            return scores

        permuted_scores = {}  # dict of feature_var:dict[phase:]
        permuted_height_scores = {}  # dict of feature_var:dict[phase:]
        # other_height_metrics = {}  # dict of feature_var:sklearn.classification_report

        assert len(CNN_FEATURE_VARS) == X_test.shape[-1]
        for i, feature_name in enumerate(tqdm(CNN_FEATURE_VARS[1:])):
            i = i + 1  # skip cloud_flag

            # Predict on permuted feature column
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, :, :, i])
            y_pred_permuted = model.predict(X_permuted, verbose=0)  # type: ignore

            # Measure accuracy by phase and by phase/height
            permuted_scores[feature_name] = phase_acc_scores(y_pred_permuted.ravel())
            permuted_height_scores[feature_name] = phase_acc_height_scores(
                y_true=y_test, y_pred=y_pred_permuted
            )

        phase_baseline = phase_acc_scores(y_pred)
        phase_height_baseline = phase_acc_height_scores(y_test, pred)

        # phase importances
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
        phase_importance_df.to_parquet(PHASE_IMPORTANCE_PATH)

        # phase-height importances
        phase_height_dfs = {}
        for phase in list(PHASES) + ["overall"]:
            df = pd.DataFrame({k: v[phase] for k, v in permuted_height_scores.items()})
            df = -df.subtract(
                phase_height_baseline[phase], axis=0
            )  # baseline-permuted = score
            df = df.reset_index(names=["height_idx"])
            df = df.melt(id_vars=["height_idx"], value_name="importance")
            df["phase"] = phase
            phase_height_dfs[phase] = df
        phase_height_importance_df = pd.concat(phase_height_dfs.values())
        phase_height_importance_df["height"] = 0.16 + (
            0.03 * phase_height_importance_df.pop("height_idx")
        )  # type: ignore
        phase_height_importance_df.to_parquet(PHASE_HEIGHT_IMPORTANCE_PATH)


def calculate_mlp_importances(
    mlp: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray
):
    PHASE_IMPORTANCE_PATH = Path("./data/importance/mlp_phase_importances.parquet")
    PHASE_HEIGHT_IMPORTANCE_PATH = Path(
        "./data/importance/mlp_phase_height_importance.parquet"
    )

    y_flat = y_test.ravel()

    pred = make_mlp_pred(mlp, X_test)  # BASELINE
    y_pred = pred.ravel()

    PHASE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    phase_idxs = {phase_id: np.where(y_flat == phase_id) for phase_id in PHASE_NUMS}
    phase_values = {phase_id: (y_test == phase_id) for phase_id in PHASE_NUMS}

    def phase_acc_scores(y_pred):
        scores: dict[str, float] = {}
        for phase_id, idxs in phase_idxs.items():
            if not len(idxs[0]):
                scores[PHASE_MAP[phase_id]] = np.nan
                continue
            scores[PHASE_MAP[phase_id]] = (y_pred[idxs] == phase_id).mean()
        scores["avg"] = np.nanmean(list(scores.values()))  # type: ignore
        return scores

    def phase_acc_height_scores(y_true, y_pred):
        # y_true/pred should have shape (samples, 128, 384)
        matches = y_true == y_pred
        scores = pd.DataFrame({"overall": matches.mean(axis=(0, 1))})
        for phase_id, of_phase_type in phase_values.items():
            numerator = (matches & of_phase_type).sum(axis=(0, 1))
            denominator = of_phase_type.sum(axis=(0, 1))
            acc_for_phase = np.full_like(numerator, fill_value=np.nan, dtype=float)
            np.divide(
                numerator, denominator, out=acc_for_phase, where=denominator > 100
            )
            scores[PHASE_MAP[phase_id]] = acc_for_phase
        scores["overall"] = matches.mean(axis=(0, 1))
        return scores

    permuted_scores = {}  # dict of feature_var:dict[phase:]
    permuted_height_scores = {}  # dict of feature_var:dict[phase:]
    # other_height_metrics = {}  # dict of feature_var:sklearn.classification_report

    feature_vars = [
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
    assert len(feature_vars) == X_test.shape[-1]
    for i, feature_name in enumerate(tqdm(feature_vars[1:])):
        i = i + 1  # skip cloud_flag

        # Predict on permuted feature column
        X_permuted = X_test.copy()
        np.random.shuffle(X_permuted[:, :, :, i])
        y_pred_permuted = make_mlp_pred(mlp, X_permuted)

        # Measure accuracy by phase and by phase/height
        permuted_scores[feature_name] = phase_acc_scores(y_pred_permuted.ravel())
        permuted_height_scores[feature_name] = phase_acc_height_scores(
            y_true=y_test, y_pred=y_pred_permuted
        )

    phase_baseline = phase_acc_scores(y_pred)
    phase_height_baseline = phase_acc_height_scores(y_test, pred)

    # phase importances
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
    phase_importance_df.to_parquet(PHASE_IMPORTANCE_PATH)

    # phase-height importances
    phase_height_dfs = {}
    for phase in list(PHASES) + ["overall"]:
        df = pd.DataFrame({k: v[phase] for k, v in permuted_height_scores.items()})
        df = -df.subtract(
            phase_height_baseline[phase], axis=0
        )  # baseline-permuted = score
        df = df.reset_index(names=["height_idx"])
        df = df.melt(id_vars=["height_idx"], value_name="importance")
        df["phase"] = phase
        phase_height_dfs[phase] = df
    phase_height_importance_df = pd.concat(phase_height_dfs.values())
    phase_height_importance_df["height"] = 0.16 + (
        0.03 * phase_height_importance_df.pop("height_idx")
    )  # type: ignore
    phase_height_importance_df.to_parquet(PHASE_HEIGHT_IMPORTANCE_PATH)


def calculate_rf_importances(rf: RandomForestClassifier):
    PHASE_IMPORTANCE_PATH = Path("./data/importance/rf_phase_importances.parquet")
    PHASE_HEIGHT_IMPORTANCE_PATH = Path(
        "./data/importance/rf_phase_height_importance.parquet"
    )

    # load data and get X & y test set
    ds = xr.open_mfdataset("../preprocessing/data/raw/nsathermocldphaseC1.c1.2021*.nc")[
        list(rf.feature_names_in_) + ["cloud_flag", "cloud_phase_mplgr"]
    ]
    df = ds.to_dataframe().fillna(0)
    df = df[df["cloud_flag"] == 1]
    ds.close()
    del ds
    df_known_phase = df[~df["cloud_phase_mplgr"].isin([0.0, 8.0])]
    X_test = df_known_phase[rf.feature_names_in_]
    y_test = df_known_phase["cloud_phase_mplgr"]

    y_pred = make_rf_pred(rf, X_test)  # BASELINE

    PHASE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    phase_idxs = {phase_id: np.where(y_test == phase_id) for phase_id in PHASE_NUMS}
    phase_values = {phase_id: (y_test == phase_id) for phase_id in PHASE_NUMS}

    def phase_acc_scores(y_pred):
        scores: dict[str, float] = {}
        for phase_id, idxs in phase_values.items():
            if not idxs.any() or idxs.sum() < 100:
                scores[PHASE_MAP[phase_id]] = np.nan
                continue
            scores[PHASE_MAP[phase_id]] = (y_pred[idxs] == phase_id).mean()
        scores["avg"] = np.nanmean(list(scores.values()))  # type: ignore
        return scores

    def phase_acc_height_scores(y_pred):
        matches = pd.Series((y_test == y_pred), name="correct")
        scores = pd.DataFrame(
            matches.reset_index()
            .groupby("height")
            .agg(overall=pd.NamedAgg(column="correct", aggfunc="mean"))
        )
        for phase_id, of_phase_type in phase_values.items():
            numerator = (matches & of_phase_type).sum()
            denominator = of_phase_type.sum()
            acc_for_phase = np.full_like(numerator, fill_value=np.nan, dtype=float)
            np.divide(
                numerator, denominator, out=acc_for_phase, where=denominator > 100
            )
            scores[PHASE_MAP[phase_id]] = acc_for_phase
        return scores

    permuted_scores = {}  # dict of feature_var:dict[phase:]
    permuted_height_scores = {}  # dict of feature_var:dict[phase:]

    for i, feature_name in enumerate(tqdm(rf.feature_names_in_)):
        # Predict on permuted feature column
        X_permuted = X_test.copy()
        X_permuted[feature_name] = X_permuted[feature_name].sample(frac=1).values
        y_pred_permuted = make_rf_pred(rf, X_permuted)

        # Measure accuracy by phase and by phase/height
        permuted_scores[feature_name] = phase_acc_scores(y_pred_permuted.ravel())
        permuted_height_scores[feature_name] = phase_acc_height_scores(
            y_pred=y_pred_permuted
        )

    phase_baseline = phase_acc_scores(y_pred)
    phase_height_baseline = phase_acc_height_scores(y_pred)

    # phase importances
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
    phase_importance_df.to_parquet(PHASE_IMPORTANCE_PATH)

    # phase-height importances
    phase_height_dfs = {}
    for phase in list(PHASES) + ["overall"]:
        df = pd.DataFrame({k: v[phase] for k, v in permuted_height_scores.items()})
        df = -df.subtract(
            phase_height_baseline[phase], axis=0
        )  # baseline-permuted = score
        df = df.reset_index(names=["height_idx"])
        df = df.melt(id_vars=["height_idx"], value_name="importance")
        df["phase"] = phase
        phase_height_dfs[phase] = df
    phase_height_importance_df = pd.concat(phase_height_dfs.values())
    phase_height_importance_df["height"] = 0.16 + (
        0.03 * phase_height_importance_df.pop("height_idx")
    )  # type: ignore
    phase_height_importance_df.to_parquet(PHASE_HEIGHT_IMPORTANCE_PATH)


def calculate_importances():
    print("loading models...")
    cnnA, cnnB, mlp, rf = load_models()

    print("loading training data for CNN and MLP models...")
    X_test, y_test = load_data()

    print("calculating cnn feature importances (this could take a while)...")
    calculate_cnn_importances(cnnA, cnnB, X_test, y_test)

    print("calculating mlp feature importances (this could take a while)...")
    calculate_mlp_importances(mlp, X_test, y_test)

    # hopeless attempt to save some memory
    del X_test
    del y_test

    print("calculating rf feature importances (this could take a while)...")
    calculate_rf_importances(rf)


if __name__ == "__main__":
    calculate_importances()
