"""
Tests for validating registry YAML data integrity.

These tests verify that the registry data files (modes and performance) contain
correct values that match the specifications they claim to implement.
"""

import pytest

from spacelink.phy.registry import Registry


class TestDVBS2Modes:
    """Validate DVB-S2 MODCOD data values."""

    @pytest.fixture(scope="class")
    def registry(self):
        """Load the registry once for all tests in this class."""
        reg = Registry()
        reg.load()
        return reg

    def test_dvbs2_spectral_efficiency(self, registry):
        """Verify DVB-S2 MODCODs match spectral efficiency from the standard.

        Reference: ETSI EN 302 307-1 Table 13.
        https://www.etsi.org/deliver/etsi_en/302300_302399/30230701/01.04.01_60/en_30230701v010401p.pdf
        """

        # Spectral efficiency values directly from Table 13.
        expected = {
            "DVBS2_QPSK_1/4": 0.490243,
            "DVBS2_QPSK_1/3": 0.656448,
            "DVBS2_QPSK_2/5": 0.789412,
            "DVBS2_QPSK_1/2": 0.988858,
            "DVBS2_QPSK_3/5": 1.188304,
            "DVBS2_QPSK_2/3": 1.322253,
            "DVBS2_QPSK_3/4": 1.487473,
            "DVBS2_QPSK_4/5": 1.587196,
            "DVBS2_QPSK_5/6": 1.654663,
            "DVBS2_QPSK_8/9": 1.766451,
            "DVBS2_QPSK_9/10": 1.788612,
            "DVBS2_8PSK_3/5": 1.779991,
            "DVBS2_8PSK_2/3": 1.980636,
            "DVBS2_8PSK_3/4": 2.228124,
            "DVBS2_8PSK_5/6": 2.478562,
            "DVBS2_8PSK_8/9": 2.646012,
            "DVBS2_8PSK_9/10": 2.679207,
            "DVBS2_16APSK_2/3": 2.637201,
            "DVBS2_16APSK_3/4": 2.966728,
            "DVBS2_16APSK_4/5": 3.165623,
            "DVBS2_16APSK_5/6": 3.300184,
            "DVBS2_16APSK_8/9": 3.523143,
            "DVBS2_16APSK_9/10": 3.567342,
            "DVBS2_32APSK_3/4": 3.703295,
            "DVBS2_32APSK_4/5": 3.951571,
            "DVBS2_32APSK_5/6": 4.119540,
            "DVBS2_32APSK_8/9": 4.397854,
            "DVBS2_32APSK_9/10": 4.453027,
        }

        for mode_id, expected_efficiency in expected.items():
            mode = registry.modes[mode_id]
            actual_efficiency = float(mode.info_bits_per_symbol)
            assert actual_efficiency == pytest.approx(expected_efficiency, abs=1e-6)
