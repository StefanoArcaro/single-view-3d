from dataclasses import dataclass

import numpy as np


@dataclass
class TemplateAnalysisData:
    """Analysis results for a single template."""

    id: str
    distance_pred: float
    distance_true: float | None = None

    @property
    def error_abs(self) -> float | None:
        """Absolute error if ground-truth is available."""
        if self.distance_true is None:
            return None
        return abs(self.distance_pred - self.distance_true)

    @property
    def error_rel(self) -> float | None:
        """Relative error if ground-truth is available."""
        if self.distance_true is None or self.distance_true == 0:
            return None
        return self.error_abs / self.distance_true


@dataclass
class PoseAnalysisData:
    """3D pose data for a single template."""

    id: str
    R: np.ndarray
    t: np.ndarray


@dataclass
class AnalysisData:
    """Analysis results for a scene."""

    scene_id: str
    units: str
    templates: list[TemplateAnalysisData]
    poses: list[PoseAnalysisData]

    def has_ground_truth(self) -> bool:
        """Check if any template has ground-truth distance."""
        return any(t.distance_true is not None for t in self.templates)

    def get_summary_stats(self) -> dict:
        """Get summary statistics for the analysis data."""
        with_gt = [t for t in self.templates if t.distance_true is not None]

        if not with_gt:
            return {
                "units": self.units,
                "total_templates": len(self.templates),
                "templates_with_gt": 0,
                "mean_error_abs": None,
                "mean_error_rel": None,
            }

        error_abs = [t.error_abs for t in with_gt]
        error_rel = [t.error_rel for t in with_gt if t.error_rel is not None]

        return {
            "units": self.units,
            "total_templates": len(self.templates),
            "templates_with_gt": len(with_gt),
            "mean_error_abs": float(np.mean(error_abs)),
            "mean_error_rel": float(np.mean(error_rel)) if error_rel else None,
            "min_error_abs": float(np.min(error_abs)),
            "max_error_abs": float(np.max(error_abs)),
            "min_error_rel": float(np.min(error_rel)) if error_rel else None,
            "max_error_rel": float(np.max(error_rel)) if error_rel else None,
        }
