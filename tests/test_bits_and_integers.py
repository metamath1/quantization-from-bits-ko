import pytest

from src.bits_and_integers import (
    decode_twos_complement,
    decode_unsigned,
    encode_twos_complement,
    encode_unsigned,
    enumerate_interpretations,
    signed_range,
    state_count,
)


def test_four_bits_have_sixteen_states() -> None:
    rows = enumerate_interpretations(4)
    assert state_count(4) == 16
    assert len(rows) == 16
    assert rows[0].bits == "0000"
    assert rows[-1].bits == "1111"


def test_unsigned_int4_range_is_zero_to_fifteen() -> None:
    rows = enumerate_interpretations(4)
    assert [row.unsigned for row in rows] == list(range(16))


def test_signed_int4_uses_twos_complement_range() -> None:
    rows = enumerate_interpretations(4)
    assert signed_range(4) == (-8, 7)
    assert [row.signed for row in rows] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        -8,
        -7,
        -6,
        -5,
        -4,
        -3,
        -2,
        -1,
    ]


def test_same_bits_can_have_different_integer_meanings() -> None:
    assert decode_unsigned("1101") == 13
    assert decode_twos_complement("1101") == -3


@pytest.mark.parametrize(
    ("value", "bits"),
    [
        (-8, "1000"),
        (-3, "1101"),
        (-1, "1111"),
        (0, "0000"),
        (7, "0111"),
    ],
)
def test_signed_int4_encoding_examples(value: int, bits: str) -> None:
    assert encode_twos_complement(value, 4) == bits
    assert decode_twos_complement(bits) == value


def test_all_signed_int4_values_round_trip() -> None:
    for value in range(-8, 8):
        bits = encode_twos_complement(value, 4)
        assert decode_twos_complement(bits) == value


def test_out_of_range_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        encode_unsigned(16, 4)
    with pytest.raises(ValueError):
        encode_twos_complement(8, 4)
    with pytest.raises(ValueError):
        encode_twos_complement(-9, 4)


def test_invalid_bit_strings_are_rejected() -> None:
    with pytest.raises(ValueError):
        decode_unsigned("")
    with pytest.raises(ValueError):
        decode_twos_complement("10a1")
