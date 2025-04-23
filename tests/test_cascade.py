import math
import pytest
import numpy as np
import yaml
import os
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from spacelink.cascade import Stage, Cascade


# Register YAML constructor for Quantity objects
def quantity_constructor(loader, node):
    """Constructor for !Quantity tags in YAML files."""
    mapping = loader.construct_mapping(node)
    value = mapping.get("value")

    # Check for different key names that could contain the unit
    unit_str = mapping.get("unit")
    if unit_str is None:
        unit_str = mapping.get("units")

    if unit_str is None:
        raise ValueError("Quantity must have 'unit' or 'units' key")

    # Handle special cases
    if unit_str == "linear":
        return float(value) * u.dimensionless_unscaled
    elif unit_str == "dB/K":
        return float(value) * u.dB / u.K
    elif unit_str == "dBW":
        # Handle dBW unit differently since u.dB(u.W) syntax may not be supported in some versions
        return float(value) * u.dBW
    else:
        return float(value) * getattr(u, unit_str)


yaml.SafeLoader.add_constructor("!Quantity", quantity_constructor)


def test_stage_gain_and_loss_exclusive():
    with pytest.raises(ValueError):
        Stage(label="bad", gain=10 * u.dB, loss=3 * u.dB)


def test_stage_gain_linear_and_loss_linear():
    s_gain = Stage(label="g", gain=10 * u.dB)
    assert_quantity_allclose(s_gain.gain, 10 * u.dB)
    # Linear gain is dimensionless
    assert_quantity_allclose(s_gain.gain_lin, 10 * u.dimensionless)
    s_loss = Stage(label="l", loss=3 * u.dB)
    assert_quantity_allclose(s_loss.gain, -3 * u.dB)
    assert_quantity_allclose(
        s_loss.gain_lin, 10 ** (-3 / 10) * u.dimensionless, rtol=1e-6
    )


def test_stage_noise_figure_and_temperature():
    # From noise_figure to noise_temp
    s_nf = Stage(label="nf", gain=0 * u.dB, noise_figure=2 * u.dB)
    expected_temp = 290.0 * (10 ** (2 / 10) - 1)
    assert_quantity_allclose(s_nf.noise_temp, expected_temp * u.K)
    # From noise_temp to noise_figure
    s_temp = Stage(label="nt", gain=0 * u.dB, noise_temp=200 * u.K)
    expected_nf = 10 * math.log10(1 + 200 / 290.0)
    assert_quantity_allclose(s_temp.noise_figure, expected_nf * u.dB, rtol=1e-6)


def test_stage_return_loss_vswr_conversion():
    s_rl = Stage(label="rl", gain=0 * u.dB, input_rl_db=10 * u.dB)
    # compute vswr from rl=10 dB
    rho = 10 ** (-10 / 20)
    expected_vswr = (1 + rho) / (1 - rho)
    assert_quantity_allclose(
        s_rl.input_vswr, expected_vswr * u.dimensionless, rtol=1e-6
    )
    # reverse: vswr to rl
    s_vswr = Stage(label="vs", gain=0 * u.dB, input_vswr=2.5 * u.dimensionless)
    rho2 = abs((2.5 - 1) / (2.5 + 1))
    expected_rl = -20 * math.log10(rho2)
    assert_quantity_allclose(s_vswr.input_rl_db, expected_rl * u.dB, rtol=1e-6)


def test_stage_to_from_dict():
    s = Stage(
        label="t",
        gain=5 * u.dB,
        noise_figure=3 * u.dB,
        p1db_dbm=15 * u.dBm,
        input_rl_db=12 * u.dB,
        output_rl_db=14 * u.dB,
    )
    d = s.to_dict()
    s2 = Stage.from_dict(d)
    assert s == s2
    # Dict should contain expected keys
    assert d["label"] == "t"
    assert np.isclose(d["gain"], 5.0)
    assert np.isclose(d["noise_figure"], 3.0)
    assert np.isclose(d["p1db_dbm"], 15.0)
    assert np.isclose(d["input_rl_db"], 12.0)
    assert np.isclose(d["output_rl_db"], 14.0)


def test_cascade_gain():
    s1 = Stage(label="a", gain=10 * u.dB)
    s2 = Stage(label="b", gain=3 * u.dB)
    c = Cascade([s1, s2])
    assert_quantity_allclose(c.cascaded_gain(), 13 * u.dB)


def test_cascade_noise_figure_and_temperature():
    # Stage1: gf=10db nf=2db, Stage2: gf=3db nf=3db
    s1 = Stage(label="a", gain=10 * u.dB, noise_figure=2 * u.dB)
    s2 = Stage(label="b", gain=10 * u.dB, noise_figure=4 * u.dB)
    s3 = Stage(label="c", gain=10 * u.dB, noise_figure=4 * u.dB)
    c = Cascade([s1, s2, s3])
    # Compare cascaded noise figure in dB
    assert_quantity_allclose(
        c.cascaded_noise_figure_db(), 2.43 * u.dB, atol=0.01 * u.dB
    )
    # Temperature
    assert_quantity_allclose(
        c.cascaded_noise_temperature_k(), 217.85 * u.K, atol=0.01 * u.K
    )
    assert_quantity_allclose(c.cascaded_gain(), 30 * u.dB)


def test_cascade_input_referred_p1db_single_stage():
    s = Stage(label="x", gain=5 * u.dB, p1db_dbm=20 * u.dBm)
    c = Cascade([s])
    # input-referred = output - gain
    # input-referred = output P1dB minus total gain
    # input-referred = output P1dB minus total gain (in dB)
    expected = 15 * u.dBm
    assert_quantity_allclose(c.input_referred_p1db_dbm(), expected)


def test_cascade_len_and_getitem():
    s1 = Stage(label="a", gain=1 * u.dB)
    s2 = Stage(label="b", gain=2 * u.dB)
    c = Cascade([s1, s2])
    assert len(c) == 2
    assert c[0] is s1
    assert c[1] is s2


def test_cascade_to_from_dict_and_yaml():
    s1 = Stage(label="a", gain=4 * u.dB, noise_figure=1 * u.dB)
    s2 = Stage(label="b", loss=2 * u.dB, noise_figure=2 * u.dB)
    c = Cascade([s1, s2])
    d = c.to_dict()
    # Round-trip dict
    c2 = Cascade.from_dict(d)
    assert c == c2
    # YAML serialization
    y = yaml.safe_dump(d)
    loaded = yaml.safe_load(y)
    c3 = Cascade.from_dict(loaded)
    assert c3 == c


def test_errors_on_empty_cascade():
    c = Cascade()
    with pytest.raises(ValueError):
        _ = c.cascaded_noise_figure_db()
    with pytest.raises(ValueError):
        _ = c.input_referred_p1db_dbm()


def test_cascade_from_yaml_single_stage():
    """Test loading a single-stage cascade from a YAML string."""
    yaml_str = (
        "stages:\n" "  - label: amp1\n" "    gain: 10.0\n" "    noise_figure: 1.5\n"
    )
    data = yaml.safe_load(yaml_str)
    c = Cascade.from_dict(data)
    expected = Cascade([Stage(label="amp1", gain=10 * u.dB, noise_figure=1.5 * u.dB)])
    assert c == expected


def test_cascade_from_yaml_file_simple():
    """Test loading a simple cascade from a YAML fixture file."""
    path = os.path.join(os.path.dirname(__file__), "test_cases", "cascade_simple.yaml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    c = Cascade.from_dict(data)
    expected = Cascade(
        [Stage(label="simple_amp", gain=10 * u.dB, noise_figure=3 * u.dB)]
    )
    assert c == expected


def test_cascade_from_yaml_file_multiple():
    """Test loading a multi-stage cascade from a YAML fixture file."""
    path = os.path.join(os.path.dirname(__file__), "test_cases", "cascade_multi.yaml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    c = Cascade.from_dict(data)
    expected = Cascade(
        [
            Stage(label="amp1", gain=6 * u.dB, noise_figure=1 * u.dB),
            Stage(label="att1", loss=2 * u.dB, noise_figure=0.5 * u.dB),
            Stage(label="amp2", gain=4 * u.dB, noise_figure=2 * u.dB),
        ]
    )
    assert c == expected


def test_cascade_vswr_fixture_file():
    """Test loading a cascade from YAML with input VSWR and verify mismatch attributes."""
    path = os.path.join(os.path.dirname(__file__), "test_cases", "cascade_vswr.yaml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    c = Cascade.from_dict(data)
    # Expect two stages
    assert len(c) == 2
    # Stage 1 properties
    assert c[0].label == "s1"
    assert pytest.approx(c[0].gain.to(u.dB).value, abs=1e-6) == 10.0
    # Stage 2 properties: gain and VSWR
    assert c[1].label == "s2"
    assert pytest.approx(c[1].gain.to(u.dB).value, abs=1e-6) == 5.0
    assert_quantity_allclose(c[1].input_vswr, 2 * u.dimensionless)
    # From VSWR, input return loss should be ~9.542 dB
    rho = (2.0 - 1.0) / (2.0 + 1.0)
    expected_rl = -20.0 * math.log10(abs(rho))
    assert_quantity_allclose(c[1].input_rl_db, expected_rl * u.dB, rtol=1e-6)

    # Total gain minus mismatch loss (~0.512 dB)
    assert_quantity_allclose(c.cascaded_gain(), (15.0 - 0.512) * u.dB, atol=0.01 * u.dB)
