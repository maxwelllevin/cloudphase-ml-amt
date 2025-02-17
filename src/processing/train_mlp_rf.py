# Trains MLP and RF models using two approaches
# 1. Trains the MLP and RF using an evenly-balanced training set sampled from the CNN
#   training dataset. (200k samples for each cloud phase class 1-7)
# 2. Trains identical MLP and RF models using the same unbalanced training set as the
#   CNN, using 1_600_000 samples in total (same total # as approach 1)
from pathlib import Path
from typing import TypeVar

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

_DATA_FOLDER = Path(__file__).parent.parent / "preprocessing/data/cnn_inputs"
X_TRAIN_PATH = _DATA_FOLDER / "X_train.npy"
Y_TRAIN_PATH = _DATA_FOLDER / "y_train.npy"

MODEL_FOLDER = Path(__file__).parent / "models"
MODEL_FOLDER.mkdir(exist_ok=True)


def create_random_forest() -> RandomForestClassifier:
    model = RandomForestClassifier(random_state=1123413)
    return model


def create_mlp() -> MLPClassifier:
    model = MLPClassifier(
        (100,) * 5,
        random_state=112341,
        n_iter_no_change=50,
        early_stopping=True,
        learning_rate="adaptive",
        validation_fraction=0.2,
    )
    return model


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    X_data = np.load(X_TRAIN_PATH)
    y_data = np.load(Y_TRAIN_PATH)

    # Remove clouds and reshape for dataframe
    X = X_data.reshape(-1, X_data.shape[-1])
    y = y_data.reshape(-1, y_data.shape[-1]).flatten()
    where_cloudy = np.where((y != 0) & (y != 8))
    X = X[where_cloudy]
    y = y[where_cloudy]

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
    phase_map = {
        1: "liquid",
        2: "ice",
        3: "mixed",
        4: "drizzle",
        5: "liq_driz",
        6: "rain",
        7: "snow",
    }

    y_series = pd.Series(
        pd.Categorical(
            pd.Series(y).map(phase_map),
            categories=list(phase_map.values()),
        )
    )
    return X_df, y_series


def get_balanced_dataset(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Loads the data for approach 1: evenly-balanced dataset, 200k samples for each
    cloud phase"""
    evenly_distributed = []
    for phase in sorted(y_train.unique()):  # 8 unique phases
        phase_data = y_train[y_train == phase]
        evenly_distributed.append(phase_data.sample(200_000, random_state=42))
    y_balance: pd.Series = pd.concat(evenly_distributed)  # type: ignore
    X_balance = X_train.loc[y_balance.index]
    return X_balance, y_balance


def get_imbalanced_dataset(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Loads the data for approach 2: using the same distribution as the CNN dataset and
    same number of samples as approach 1."""
    # 1.6m is enough to match the distribution pretty closely
    y_imbalance = y_train.sample(1_600_000, random_state=42)
    X_imbalance = X_train.loc[y_imbalance.index]
    return X_imbalance, y_imbalance


Model = TypeVar("Model", MLPClassifier, RandomForestClassifier)


def train(model: Model, X: pd.DataFrame, y: pd.Series) -> Model:
    model = model.fit(X, y)
    return model


def save(model: Model, label: str) -> None:
    model_path = MODEL_FOLDER / f"{label}.joblib"
    joblib.dump(model, model_path)
    print(f"Wrote model file to {model_path}")


def main():
    print("Loading data...")
    X_train, y_train = load_training_data()

    print("Loading balanced dataset (approach 1.)")
    X_balance, y_balance = get_balanced_dataset(X_train, y_train)

    print("Training MLP and RF models using balanced dataset...")
    print("\ttraining mlp...")
    mlp_balanced = train(create_mlp(), X_balance, y_balance)
    save(mlp_balanced, "mlp_balanced")
    print("\ttraining rf...")
    rf_balanced = train(create_random_forest(), X_balance, y_balance)
    save(rf_balanced, "rf_balanced")

    del X_balance  # saves some memory
    del y_balance

    print("Loading imbalanced dataset (approach 2.)")
    X_imbalance, y_imbalance = get_imbalanced_dataset(X_train, y_train)

    print("Training MLP and RF models using imbalanced dataset...")
    print("\ttraining mlp...")
    mlp_imbalanced = train(create_mlp(), X_imbalance, y_imbalance)
    save(mlp_imbalanced, "mlp_imbalanced")
    print("\ttraining rf...")
    rf_imbalanced = train(create_random_forest(), X_imbalance, y_imbalance)
    save(rf_imbalanced, "rf_imbalanced")

    print("done!")


if __name__ == "__main__":
    main()
