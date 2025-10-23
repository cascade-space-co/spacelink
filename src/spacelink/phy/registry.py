from pathlib import Path

import yaml

from spacelink.phy.mode import LinkMode
from spacelink.phy.performance import (
    ErrorMetric,
    ModePerformanceCurve,
    ModePerformanceThreshold,
)

MODES_DIR = Path(__file__).parent / "data/modes"
PERF_DIR = Path(__file__).parent / "data/perf"


class DuplicateRegistryEntryError(Exception):
    """Raised when duplicate entries are found during registry loading."""


class NoRegistryFilesError(Exception):
    """Raised when no YAML files are found in the specified directories."""


class Registry:
    r"""
    Registry of link modes and their performance.
    """

    def __init__(self):
        r"""
        Create an empty registry.
        """
        self.modes: dict[str, LinkMode] = {}
        self.curves: list[ModePerformanceCurve] = []
        self.thresholds: list[ModePerformanceThreshold] = []
        self.curve_index: dict[tuple[str, ErrorMetric], ModePerformanceCurve] = {}
        self.threshold_index: dict[
            tuple[str, ErrorMetric], ModePerformanceThreshold
        ] = {}

    def load(
        self, mode_dir: Path = MODES_DIR, perf_dir: Path | None = PERF_DIR
    ) -> None:
        r"""
        Load link modes and performance data from files.

        Parameters
        ----------
        mode_dir : Path
            Path to the directory containing the link mode files.
        perf_dir : Path | None, optional
            Path to the directory containing the performance data. If None, no
            performance data will be loaded.

        Raises
        ------
        DuplicateRegistryEntryError
            If duplicate entries are found during loading.
        NoRegistryFilesError
            If no YAML files are found in the specified directories.
        """
        self._load_modes(mode_dir)

        if perf_dir is None:
            return

        perf_files = list(perf_dir.glob("*.yaml"))

        if not perf_files:
            raise NoRegistryFilesError(
                f"No YAML files found in performance directory '{perf_dir}'"
            )

        for file in perf_files:
            self._load_performance_file(file)

    def get_performance_curve(
        self, mode_id: str, metric: ErrorMetric
    ) -> ModePerformanceCurve:
        r"""
        Get performance curve data for a mode.

        Parameters
        ----------
        mode_id : str
            ID of the link mode.
        metric : ErrorMetric
            Error metric.

        Returns
        -------
        ModePerformanceCurve
            Performance curve object with multiple data points for interpolation.

        Raises
        ------
        KeyError
            If no curve data is available for the specified mode and metric.
        """
        return self.curve_index[(mode_id, metric)]

    def get_performance_threshold(
        self, mode_id: str, metric: ErrorMetric
    ) -> ModePerformanceThreshold:
        r"""
        Get performance threshold data for a mode.

        Parameters
        ----------
        mode_id : str
            ID of the link mode.
        metric : ErrorMetric
            Error metric.

        Returns
        -------
        ModePerformanceThreshold
            Performance threshold object with single quasi-error-free operating point.

        Raises
        ------
        KeyError
            If no threshold data is available for the specified mode and metric.
        """
        return self.threshold_index[(mode_id, metric)]

    def get_performance(
        self, mode_id: str, metric: ErrorMetric
    ) -> ModePerformanceCurve:
        r"""
        Get performance curve data for a mode.

        .. deprecated:: 0.2.0
            Use :meth:`get_performance_curve` or
            :meth:`get_performance_threshold` instead. This method only
            returns curve data and will raise KeyError for threshold data.

        Parameters
        ----------
        mode_id : str
            ID of the link mode.
        metric : ErrorMetric
            Error metric.

        Returns
        -------
        ModePerformanceCurve
            Performance curve object.

        Raises
        ------
        KeyError
            If no curve data is available for the specified mode and metric.
        """
        import warnings

        warnings.warn(
            "get_performance() is deprecated, use get_performance_curve() or "
            "get_performance_threshold() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_performance_curve(mode_id, metric)

    def _load_modes(self, mode_dir: Path) -> None:
        """Load link modes from YAML files."""
        mode_files = list(mode_dir.glob("*.yaml"))

        if not mode_files:
            raise NoRegistryFilesError(
                f"No YAML files found in mode directory '{mode_dir}'"
            )

        for file in mode_files:
            with open(file) as f:
                raw = yaml.safe_load(f)
                for entry in raw:
                    mode = LinkMode(**entry)
                    if mode.id in self.modes:
                        raise DuplicateRegistryEntryError(
                            f"Duplicate mode ID '{mode.id}' found"
                        )
                    self.modes[mode.id] = mode

    def _load_performance_curve(
        self, raw: dict, mode_ids: list[str], metric: ErrorMetric
    ) -> None:
        """Load curve performance data from parsed YAML."""
        perf = ModePerformanceCurve(
            modes=[self.modes[mode_id] for mode_id in mode_ids],
            metric=metric,
            points=raw["points"],
            ref=raw.get("ref", ""),
        )
        self.curves.append(perf)

        for mode_id in mode_ids:
            key = (mode_id, metric)
            if key in self.curve_index:
                raise DuplicateRegistryEntryError(
                    f"Duplicate curve performance entry for mode "
                    f"'{mode_id}' and metric '{metric.value}' found"
                )
            self.curve_index[key] = perf

    def _load_performance_threshold(
        self, raw: dict, mode_ids: list[str], metric: ErrorMetric
    ) -> None:
        """Load threshold performance data from parsed YAML."""
        threshold_data = raw["threshold"]
        perf = ModePerformanceThreshold(
            modes=[self.modes[mode_id] for mode_id in mode_ids],
            metric=metric,
            ebn0=threshold_data["ebn0"],
            error_rate=threshold_data["error_rate"],
            ref=raw.get("ref", ""),
        )
        self.thresholds.append(perf)

        for mode_id in mode_ids:
            key = (mode_id, metric)
            if key in self.threshold_index:
                raise DuplicateRegistryEntryError(
                    f"Duplicate threshold performance entry for "
                    f"mode '{mode_id}' and metric '{metric.value}' found"
                )
            self.threshold_index[key] = perf

    def _load_performance_file(self, file: Path) -> None:
        """Load a single performance YAML file."""
        with open(file) as f:
            raw = yaml.safe_load(f)
            mode_ids = raw["mode_ids"]
            metric = ErrorMetric(raw["metric"])

            # Auto-detect performance type based on YAML structure
            has_points = "points" in raw
            has_threshold = "threshold" in raw

            if has_points and has_threshold:
                raise ValueError(
                    f"Invalid YAML in {file}: cannot have both 'points' and 'threshold'"
                )
            elif has_points:
                self._load_performance_curve(raw, mode_ids, metric)
            elif has_threshold:
                self._load_performance_threshold(raw, mode_ids, metric)
            else:
                raise ValueError(
                    f"Invalid YAML in {file}: must have either 'points' or 'threshold'"
                )
