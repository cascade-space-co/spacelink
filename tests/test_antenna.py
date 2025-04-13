"""Tests for the antenna module."""

import pytest
import numpy as np
from pyradio.antenna import dish_gain, dish_3db_beamwidth


def test_dish_gain():
    """Test gain calculation at various frequencies.

    Test values generated from: https://www.keysight.com/used/us/en/knowledge/calculators/antenna-gain-calculator
    """

    assert dish_gain(20, 8.4e9) == pytest.approx(63.04, abs=0.01)
    assert dish_3db_beamwidth(20, 8.4e9) == pytest.approx(0.12, abs=0.01)

    # Test at S-band (2.4 GHz)
    assert dish_gain(20, 2.4e9) == pytest.approx(52.16, abs=0.01)
    assert dish_3db_beamwidth(20, 2.4e9) == pytest.approx(0.44, abs=0.01)

    # Test invalid efficiency
    with pytest.raises(ValueError):
        dish_gain(1.0, 2.4e9, efficiency=-0.1)
    with pytest.raises(ValueError):
        dish_gain(1.0, 2.4e9, efficiency=1.1)

    # Test invalid diameter
    with pytest.raises(ValueError):
        dish_gain(-1.0, 2.4e9)
    with pytest.raises(ValueError):
        dish_gain(0.0, 2.4e9)

    # Test invalid frequency
    with pytest.raises(ValueError):
        dish_gain(1.0, -2.4e9)
    with pytest.raises(ValueError):
        dish_gain(1.0, 0.0)

    # Test invalid diameter
    with pytest.raises(ValueError):
        dish_3db_beamwidth(-1.0, 2.4e9)
    with pytest.raises(ValueError):
        dish_3db_beamwidth(0.0, 2.4e9)

    # Test invalid frequency
    with pytest.raises(ValueError):
        dish_3db_beamwidth(1.0, -2.4e9)
    with pytest.raises(ValueError):
        dish_3db_beamwidth(1.0, 0.0)
