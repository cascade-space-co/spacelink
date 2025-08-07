import yaml
from spacelink.phy.mode import LinkMode
from spacelink.phy.performance import ModePerformance, DecoderProfile, ErrorMetric
from pathlib import Path


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
        self.perf_index: dict[
            tuple[str, DecoderProfile, ErrorMetric], ModePerformance
        ] = {}

    def load(self, mode_dir: Path, perf_dir: Path) -> None:
        r"""
        Load link modes and performance data from files.

        Parameters
        ----------
        mode_dir : Path
            Path to the directory containing the link mode files.
        perf_dir : Path
            Path to the directory containing the performance data.
        """
        for file in mode_dir.glob("*.yaml"):
            with open(file) as f:
                raw = yaml.safe_load(f)
                for entry in raw:
                    mode = LinkMode(**entry)
                    self.modes[mode.id] = mode

        for file in perf_dir.glob("*.yaml"):
            with open(file) as f:
                raw = yaml.safe_load(f)
                mode_ids = raw["mode_ids"]
                decoder_profile = DecoderProfile(**raw["decoder_profile"])
                metric = ErrorMetric(raw["metric"])

                perf = ModePerformance(
                    modes=[self.modes[mode_id] for mode_id in mode_ids],
                    decoder_profile=decoder_profile,
                    metric=metric,
                    points=raw["points"],
                    ref=raw.get("ref", ""),
                )
                self.perfs.append(perf)

                for mode_id in mode_ids:
                    key = (mode_id, decoder_profile, metric)
                    self.perf_index[key] = perf

    def get_performance(
        self, mode_id: str, decoder_profile: DecoderProfile, metric: ErrorMetric
    ) -> ModePerformance:
        r"""
        Look up the performance object for a given mode, decoder profile, and metric.

        Parameters
        ----------
        mode_id : str
            ID of the link mode.
        decoder_profile : DecoderProfile
            Decoder profile.
        metric : ErrorMetric
            Error metric.

        Returns
        -------
        ModePerformance
            Performance object.
        """
        return self.perf_index[(mode_id, decoder_profile, metric)]
