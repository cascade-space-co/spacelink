import yaml
from spacelink.phy.mode import LinkMode
from spacelink.phy.performance import ModePerformance, ErrorMetric
from pathlib import Path

MODES_DIR = Path(__file__).parent / "data/modes"
PERF_DIR = Path(__file__).parent / "data/perf"


class DuplicateRegistryEntryError(Exception):
    """Raised when duplicate entries are found during registry loading."""


class Registry:
    r"""
    Registry of link modes and their performance.
    """

    def __init__(self):
        r"""
        Create an empty registry.
        """
        self.modes: dict[str, LinkMode] = {}
        self.perfs: list[ModePerformance] = []
        self.perf_index: dict[tuple[str, ErrorMetric], ModePerformance] = {}

    def load(self, mode_dir: Path = MODES_DIR, perf_dir: Path = PERF_DIR) -> None:
        r"""
        Load link modes and performance data from files.

        Parameters
        ----------
        mode_dir : Path
            Path to the directory containing the link mode files.
        perf_dir : Path
            Path to the directory containing the performance data.

        Raises
        ------
        DuplicateRegistryEntryError
            If duplicate entries are found during loading.
        """
        for file in mode_dir.glob("*.yaml"):
            with open(file) as f:
                raw = yaml.safe_load(f)
                for entry in raw:
                    mode = LinkMode(**entry)
                    if mode.id in self.modes:
                        raise DuplicateRegistryEntryError(
                            f"Duplicate mode ID '{mode.id}' found"
                        )
                    self.modes[mode.id] = mode

        for file in perf_dir.glob("*.yaml"):
            with open(file) as f:
                raw = yaml.safe_load(f)
                mode_ids = raw["mode_ids"]
                metric = ErrorMetric(raw["metric"])

                perf = ModePerformance(
                    modes=[self.modes[mode_id] for mode_id in mode_ids],
                    metric=metric,
                    points=raw["points"],
                    ref=raw.get("ref", ""),
                )
                self.perfs.append(perf)

                for mode_id in mode_ids:
                    key = (mode_id, metric)
                    if key in self.perf_index:
                        raise DuplicateRegistryEntryError(
                            f"Duplicate performance entry for mode '{mode_id}' "
                            f"and metric '{metric.value}' found"
                        )
                    self.perf_index[key] = perf

    def get_performance(self, mode_id: str, metric: ErrorMetric) -> ModePerformance:
        r"""
        Look up the performance object for a given mode and metric.

        Parameters
        ----------
        mode_id : str
            ID of the link mode.
        metric : ErrorMetric
            Error metric.

        Returns
        -------
        ModePerformance
            Performance object.
        """
        return self.perf_index[(mode_id, metric)]
