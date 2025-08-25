from fractions import Fraction

import astropy.units as u

from spacelink.phy.mode import (
    Code,
    CodeChain,
    LinkMode,
    Modulation,
)


class TestCodeChain:
    def test_rate_property_empty_codes(self):
        """Test that CodeChain.rate returns 1 when no codes are provided."""
        code_chain = CodeChain(codes=[])
        assert code_chain.rate == Fraction(1)

    def test_rate_property_single_code(self):
        """Test that CodeChain.rate returns the code rate for a single code."""
        code = Code(name="Test Code", rate=Fraction(1, 2))
        code_chain = CodeChain(codes=[code])
        assert code_chain.rate == Fraction(1, 2)

    def test_rate_property_multiple_codes(self):
        """Test that CodeChain.rate multiplies rates for multiple codes."""
        code1 = Code(name="Outer Code", rate=Fraction(1, 2))
        code2 = Code(name="Inner Code", rate=Fraction(3, 4))
        code_chain = CodeChain(codes=[code1, code2])

        # Rate should be product: 1/2 * 3/4 = 3/8
        expected_rate = Fraction(1, 2) * Fraction(3, 4)
        assert code_chain.rate == expected_rate
        assert code_chain.rate == Fraction(3, 8)


class TestLinkMode:
    def test_info_bits_per_symbol_property(self):
        # Test with BPSK (1 bit/symbol) and rate 1/2 code
        modulation = Modulation(name="BPSK", bits_per_symbol=1)
        code = Code(name="Test Code", rate=Fraction(1, 2))
        coding = CodeChain(codes=[code])
        link_mode = LinkMode(id="TEST", modulation=modulation, coding=coding)

        # Expected: 1 bit/symbol * 1/2 rate = 1/2 info bits per symbol
        assert link_mode.info_bits_per_symbol == Fraction(1, 2)

        # Test with QPSK (2 bits/symbol) and rate 3/4 code
        modulation_qpsk = Modulation(name="QPSK", bits_per_symbol=2)
        code_34 = Code(name="Test Code 3/4", rate=Fraction(3, 4))
        coding_34 = CodeChain(codes=[code_34])
        link_mode_qpsk = LinkMode(
            id="QPSK_TEST", modulation=modulation_qpsk, coding=coding_34
        )

        # Expected: 2 bits/symbol * 3/4 rate = 3/2 info bits per symbol
        assert link_mode_qpsk.info_bits_per_symbol == Fraction(3, 2)

        # Test with uncoded (rate 1)
        uncoded = CodeChain(codes=[])  # Empty codes = rate 1
        link_mode_uncoded = LinkMode(
            id="UNCODED_TEST", modulation=modulation_qpsk, coding=uncoded
        )

        # Expected: 2 bits/symbol * 1 rate = 2 info bits per symbol
        assert link_mode_uncoded.info_bits_per_symbol == Fraction(2, 1)

    def test_channel_bits_per_symbol_property(self):
        modulation_bpsk = Modulation(name="BPSK", bits_per_symbol=1)
        code = Code(name="Test Code", rate=Fraction(1, 2))
        coding = CodeChain(codes=[code])
        link_mode_bpsk = LinkMode(
            id="BPSK_TEST", modulation=modulation_bpsk, coding=coding
        )
        assert link_mode_bpsk.channel_bits_per_symbol == 1

        modulation_qpsk = Modulation(name="QPSK", bits_per_symbol=2)
        link_mode_qpsk = LinkMode(
            id="QPSK_TEST", modulation=modulation_qpsk, coding=coding
        )
        assert link_mode_qpsk.channel_bits_per_symbol == 2

        modulation_8psk = Modulation(name="8PSK", bits_per_symbol=3)
        uncoded = CodeChain(codes=[])
        link_mode_8psk = LinkMode(
            id="8PSK_TEST", modulation=modulation_8psk, coding=uncoded
        )
        assert link_mode_8psk.channel_bits_per_symbol == 3

    def test_info_bit_rate_method(self):
        # Test with BPSK (1 bit/symbol) and rate 1/2 code
        modulation = Modulation(name="BPSK", bits_per_symbol=1)
        code = Code(name="Test Code", rate=Fraction(1, 2))
        coding = CodeChain(codes=[code])
        link_mode = LinkMode(id="TEST", modulation=modulation, coding=coding)

        # Symbol rate: 1000 Hz, info bits per symbol: 1 * 1/2 = 0.5
        # Expected info bit rate: 1000 * 0.5 = 500 Hz
        symbol_rate = 1000.0 * u.Hz
        info_bit_rate = link_mode.info_bit_rate(symbol_rate)
        assert info_bit_rate == 500.0 * u.Hz

        # Test with QPSK (2 bits/symbol) and rate 3/4 code
        modulation_qpsk = Modulation(name="QPSK", bits_per_symbol=2)
        code_34 = Code(name="Test Code 3/4", rate=Fraction(3, 4))
        coding_34 = CodeChain(codes=[code_34])
        link_mode_qpsk = LinkMode(
            id="QPSK_TEST", modulation=modulation_qpsk, coding=coding_34
        )

        # Symbol rate: 2000 Hz, info bits per symbol: 2 * 3/4 = 1.5
        # Expected info bit rate: 2000 * 1.5 = 3000 Hz
        symbol_rate_qpsk = 2000.0 * u.Hz
        info_bit_rate_qpsk = link_mode_qpsk.info_bit_rate(symbol_rate_qpsk)
        assert info_bit_rate_qpsk == 3000.0 * u.Hz

    def test_symbol_rate_method(self):
        # Test with BPSK (1 bit/symbol) and rate 1/2 code
        modulation = Modulation(name="BPSK", bits_per_symbol=1)
        code = Code(name="Test Code", rate=Fraction(1, 2))
        coding = CodeChain(codes=[code])
        link_mode = LinkMode(id="TEST", modulation=modulation, coding=coding)

        # Info bit rate: 500 Hz, info bits per symbol: 1 * 1/2 = 0.5
        # Expected symbol rate: 500 / 0.5 = 1000 Hz
        info_bit_rate = 500.0 * u.Hz
        symbol_rate = link_mode.symbol_rate(info_bit_rate)
        assert symbol_rate == 1000.0 * u.Hz

        # Test with QPSK (2 bits/symbol) and rate 3/4 code
        modulation_qpsk = Modulation(name="QPSK", bits_per_symbol=2)
        code_34 = Code(name="Test Code 3/4", rate=Fraction(3, 4))
        coding_34 = CodeChain(codes=[code_34])
        link_mode_qpsk = LinkMode(
            id="QPSK_TEST", modulation=modulation_qpsk, coding=coding_34
        )

        # Info bit rate: 3000 Hz, info bits per symbol: 2 * 3/4 = 1.5
        # Expected symbol rate: 3000 / 1.5 = 2000 Hz
        info_bit_rate_qpsk = 3000.0 * u.Hz
        symbol_rate_qpsk = link_mode_qpsk.symbol_rate(info_bit_rate_qpsk)
        assert symbol_rate_qpsk == 2000.0 * u.Hz
