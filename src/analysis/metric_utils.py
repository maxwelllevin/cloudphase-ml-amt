from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

METRIC_NAMES = Literal["precision", "recall", "f1", "iou", "acc"]


class ConfusionMatrixMetrics:
    def __init__(
        self,
        confusion_matrix: pd.DataFrame | None = None,
        pred: pd.Series | None = None,
        truth: pd.Series | None = None,
        compute_immediately: bool = True,
        precision_thresh: int = 1,
        recall_thresh: int = 1,
        iou_thresh: int = 1,
    ):
        if confusion_matrix is None:
            assert pred is not None and truth is not None
            confusion_matrix = pd.crosstab(truth, pred, dropna=False)

        self.matrix = confusion_matrix
        self.n_samples = confusion_matrix.sum().sum()
        self.classes: list[str] = list(confusion_matrix.index.unique())

        self._scores: dict[METRIC_NAMES, dict[str, float] | float] = {}
        if compute_immediately:
            self._compute_scores(
                precision_thresh=precision_thresh,
                recall_thresh=recall_thresh,
                iou_thresh=iou_thresh,
            )

    def __repr__(self) -> str:
        return repr(self._scores)

    def _recall(self, col: str, thresh: int = 1) -> float:
        """Calculates recall score for the given column. Recall is the number of true
        positives divided by true positives and false negatives.

        if the divisor (true positives + false negatives) is less than thresh the result
        is set to np.nan"""
        tp = self.matrix.loc[col][col]
        tp_fn = self.matrix.loc[col].sum().item()  # by index (ground truth)
        if tp_fn < thresh:
            return np.nan
        return tp / tp_fn

    def _precision(self, col: str, thresh: int = 1) -> float:
        """Calculates precision score for the given column. Precision is the number of
        true positives divided by true positives and false positives.

        if the divisor (true positives + false positives) is less than thresh the result
        is set to np.nan"""
        tp = self.matrix.loc[col][col]
        tp_fp = self.matrix[col].sum().item()  # by column (pred)
        if tp_fp < thresh:
            return np.nan
        return tp / tp_fp

    def _f1(self, col: str, recall_thresh: int = 1, prec_thresh: int = 1) -> float:
        """Calculates f1 score for a given column. F1 score is the harmonic mean of
        precision and recall and is calculated as 2pr/(p+r)."""
        p = self._precision(col, thresh=prec_thresh)
        r = self._recall(col, thresh=recall_thresh)
        if np.isnan(p) or np.isnan(r) or (p + r) == 0:
            return np.nan
        return (2 * p * r) / (p + r)

    def _iou(self, col: str, thresh: int = 1) -> float:
        """Calculates iou score for a given column. IoU is true positives divided by
        true positives plus false positives plus false negatives."""
        tp = self.matrix.loc[col][col]
        tp_fp_fn = (
            self.matrix[col].sum().item() + self.matrix.loc[col].sum().item() - tp
        )
        if tp_fp_fn < thresh:
            return np.nan
        return tp / tp_fp_fn

    def _acc(self) -> float:
        """Calculates accuracy score using the confusion matrix. This metric is only
        computed for the whole matrix, not for an individual column."""
        tp = np.diag(self.matrix).sum()
        tp_fp_fn = self.matrix.sum().sum()
        return tp / tp_fp_fn

    def _compute_scores(
        self, recall_thresh: int = 1, precision_thresh: int = 1, iou_thresh: int = 1
    ) -> None:
        self._scores["recall"] = {
            c: self._recall(c, recall_thresh) for c in self.classes
        }
        self._scores["precision"] = {
            c: self._precision(c, precision_thresh) for c in self.classes
        }
        self._scores["f1"] = {
            c: self._f1(c, recall_thresh, precision_thresh) for c in self.classes
        }
        self._scores["iou"] = {c: self._iou(c, iou_thresh) for c in self.classes}
        self._scores["acc"] = self._acc()

    @staticmethod
    def apply_weights(
        scores: dict[str, float], weights: dict[str, float] | None = None
    ) -> float:
        scores = {k: v for k, v in scores.items() if not np.isnan(v)}
        if not len(scores):
            return np.nan
        if weights is None:
            weights = {k: 1 / len(scores) for k in scores}
        weights = {k: weights[k] for k in scores}

        score_arr = np.array(list(scores.values()))
        weight_arr = np.array(list(weights.values()))
        weight_arr /= weight_arr.sum()  # ensure adds up to 1
        return (score_arr * weight_arr).sum()

    def scores(self, weights: dict[str, float] | None = None) -> pd.Series:
        if self._scores == {}:
            self._compute_scores()  # using default thresholds

        weighted_scores: dict[METRIC_NAMES, float] = {}
        for label, class_scores in self._scores.items():
            if label == "acc":  # acc is the only one that is a scalar; can't weight it
                weighted_scores[label] = class_scores  # type: ignore
            else:
                weighted_scores[label] = self.apply_weights(class_scores, weights)  # type: ignore
        return pd.Series(weighted_scores).T

    def get_matrix(
        self, norm: Literal["none", "index", "columns"] = "none"
    ) -> pd.DataFrame:
        if norm == "none":
            matrix = self.matrix
        elif norm == "index":
            matrix = self.matrix.apply(lambda row: row / row.sum(), axis=1)
        else:
            matrix = self.matrix.apply(lambda row: row / row.sum(), axis=0)
        return matrix  # type: ignore

    def plot(
        self,
        ax: plt.Axes | None = None,
        norm: Literal["none", "index", "columns"] = "none",
        pred_label: str | None = None,
    ) -> plt.Axes:
        matrix = self.get_matrix(norm=norm)

        if ax is None:
            ax = plt.axes()

        sns.heatmap(
            matrix,
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot=True,
            fmt=".2f",
            square=True,
        )
        ax.set_aspect("equal")  # makes it square
        for text in ax.texts:
            if float(text.get_text()) < 0.001:
                text.set_text("")
                text.set_color("white")
        for spine in ax.spines.values():
            spine.set_visible(True)

        if pred_label is not None:
            ax.set_xlabel(pred_label)

        return ax
