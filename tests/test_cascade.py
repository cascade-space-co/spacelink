import math
import pytest
from pint.testing import assert_allclose
import yaml

from pyradio.units import Q_, dB, dimensionless, K, dBm
from pyradio.cascade import Stage, Cascade
import os


def test_stage_gain_and_loss_exclusive():
    with pytest.raises(ValueError):
        Stage(label="bad", gain=Q_(10, dB), loss_db=Q_(3, dB))


def test_stage_gain_linear_and_loss_linear():
    s_gain = Stage(label="g", gain=Q_(10, dB))
    assert_allclose(s_gain.gain, Q_(10, dB))
    # Linear gain is dimensionless
    assert_allclose(s_gain.gain_lin, Q_(10, dimensionless))
    s_loss = Stage(label="l", loss_db=Q_(3, dB))
    assert_allclose(s_loss.gain, Q_(-3, dB))
    assert_allclose(s_loss.gain_lin, 10 ** (-3 / 10) * dimensionless, rtol=1e-6)


def test_stage_noise_figure_and_temperature():
    # From noise_figure to noise_temp
    s_nf = Stage(label="nf", gain=Q_(0, dB), noise_figure=Q_(2, dB))
    expected_temp = 290.0 * (10 ** (2 / 10) - 1)
    assert_allclose(s_nf.noise_temp, Q_(expected_temp, K))
    # From noise_temp to noise_figure
    s_temp = Stage(label="nt", gain=Q_(0, dB), noise_temp=Q_(200, K))
    expected_nf = 10 * math.log10(1 + 200 / 290.0)
    assert_allclose(s_temp.noise_figure.to(dB), expected_nf * dB, rtol=1e-6)


def test_stage_return_loss_vswr_conversion():
    s_rl = Stage(label="rl", gain=Q_(0, dB), input_rl_db=Q_(10, dB))
    # compute vswr from rl=10 dB
    rho = 10 ** (-10 / 20)
    expected_vswr = (1 + rho) / (1 - rho)
    assert_allclose(s_rl.input_vswr, expected_vswr * dimensionless, rtol=1e-6)
    # reverse: vswr to rl
    s_vswr = Stage(label="vs", gain=Q_(0, dB), input_vswr=Q_(2.5, dimensionless))
    rho2 = abs((2.5 - 1) / (2.5 + 1))
    expected_rl = -20 * math.log10(rho2)
    assert_allclose(Q_(s_vswr.input_rl_db, dB), expected_rl * dB, rtol=1e-6)


def test_stage_to_from_dict():
    s = Stage(
        label="t",
        gain=Q_(5, dB),
        noise_figure=Q_(3, dB),
        p1db_dbm=Q_(15, dBm),
        input_rl_db=Q_(12, dB),
        output_rl_db=Q_(14, dB),
    )
    d = s.to_dict()
    s2 = Stage.from_dict(d)
    assert s == s2
    # Dict should contain expected keys
    assert d["label"] == "t"
    assert_allclose(Q_(d["gain"], dB), 5 * dB)
    assert_allclose(Q_(d["noise_figure"], dB), 3 * dB)
    assert_allclose(Q_(d["p1db_dbm"], dBm), 15 * dBm)
    assert_allclose(Q_(d["input_rl_db"], dB), 12 * dB)
    assert_allclose(Q_(d["output_rl_db"], dB), 14 * dB)


def test_cascade_gain():
    s1 = Stage(label="a", gain=Q_(10, dB))
    s2 = Stage(label="b", gain=Q_(3, dB))
    c = Cascade([s1, s2])
    assert_allclose(c.cascaded_gain(), Q_(13, dB))


def test_cascade_noise_figure_and_temperature():
    # Stage1: gf=10db nf=2db, Stage2: gf=3db nf=3db
    s1 = Stage(label="a", gain=Q_(10, dB), noise_figure=Q_(2, dB))
    s2 = Stage(label="b", gain=Q_(10, dB), noise_figure=Q_(4, dB))
    s3 = Stage(label="c", gain=Q_(10, dB), noise_figure=Q_(4, dB))
    c = Cascade([s1, s2, s3])
    # Compare cascaded noise figure in dB
    assert_allclose(c.cascaded_noise_figure_db().to(dB), 2.43 * dB, atol=0.01)
    # Temperature
    assert_allclose(c.cascaded_noise_temperature_k(), Q_(217.85, K), atol=0.01)
    assert_allclose(c.cascaded_gain(), Q_(30, dB))


def test_cascade_input_referred_p1db_single_stage():
    s = Stage(label="x", gain=Q_(5, dB), p1db_dbm=Q_(20, dBm))
    c = Cascade([s])
    # input-referred = output - gain
    # input-referred = output P1dB minus total gain
    # input-referred = output P1dB minus total gain (in dB)
    expected = Q_(20 - 5, dBm)
    assert_allclose(c.input_referred_p1db_dbm(), expected)


def test_cascade_len_and_getitem():
    s1 = Stage(label="a", gain=Q_(1, dB))
    s2 = Stage(label="b", gain=Q_(2, dB))
    c = Cascade([s1, s2])
    assert len(c) == 2
    assert c[0] is s1
    assert c[1] is s2


def test_cascade_to_from_dict_and_yaml():
    s1 = Stage(label="a", gain=Q_(4, dB), noise_figure=Q_(1, dB))
    s2 = Stage(label="b", loss_db=Q_(2, dB), noise_figure=Q_(2, dB))
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
    expected = Cascade([Stage(label="amp1", gain=Q_(10, dB), noise_figure=Q_(1.5, dB))])
    assert c == expected


def test_cascade_from_yaml_file_simple():
    """Test loading a simple cascade from a YAML fixture file."""
    path = os.path.join(os.path.dirname(__file__), "test_cases", "cascade_simple.yaml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    c = Cascade.from_dict(data)
    expected = Cascade(
        [Stage(label="simple_amp", gain=Q_(10, dB), noise_figure=Q_(3, dB))]
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
            Stage(label="amp1", gain=Q_(6, dB), noise_figure=Q_(1, dB)),
            Stage(label="att1", loss_db=Q_(2, dB), noise_figure=Q_(0.5, dB)),
            Stage(label="amp2", gain=Q_(4, dB), noise_figure=Q_(2, dB)),
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
    assert pytest.approx(c[0].gain.to(dB).magnitude, abs=1e-6) == 10.0
    # Stage 2 properties: gain and VSWR
    assert c[1].label == "s2"
    assert pytest.approx(c[1].gain.to(dB).magnitude, abs=1e-6) == 5.0
    assert c[1].input_vswr == Q_(2, dimensionless)
    # From VSWR, input return loss should be ~9.542 dB
    rho = (2.0 - 1.0) / (2.0 + 1.0)
    expected_rl = -20.0 * math.log10(abs(rho))
    assert pytest.approx(c[1].input_rl_db, abs=1e-6) == expected_rl

    # Total gain minus mismatch loss (~0.512 dB)
    assert_allclose(c.cascaded_gain(), Q_(15.0 - 0.512, dB), atol=0.01)
