"""고정 비트 폭의 unsigned 정수와 2의 보수 signed 정수 예제.

Python의 ``int``는 임의 정밀도 정수이므로, 이 모듈은 교육 목적으로
비트 폭을 명시해 고정 폭 정수의 인코딩과 디코딩을 재현한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegerInterpretation:
    """하나의 비트열을 unsigned와 signed로 해석한 결과."""

    bits: str
    unsigned: int
    signed: int


def _validate_bit_width(bit_width: int) -> None:
    if isinstance(bit_width, bool) or not isinstance(bit_width, int):
        raise TypeError("bit_width는 정수여야 합니다.")
    if bit_width < 1:
        raise ValueError("bit_width는 1 이상이어야 합니다.")


def _validate_bits(bits: str) -> None:
    if not isinstance(bits, str):
        raise TypeError("bits는 문자열이어야 합니다.")
    if not bits:
        raise ValueError("bits는 비어 있을 수 없습니다.")
    if any(bit not in "01" for bit in bits):
        raise ValueError("bits에는 '0'과 '1'만 사용할 수 있습니다.")


def state_count(bit_width: int) -> int:
    """주어진 비트 수로 만들 수 있는 서로 다른 비트열의 개수를 반환한다."""
    _validate_bit_width(bit_width)
    return 1 << bit_width


def signed_range(bit_width: int) -> tuple[int, int]:
    """2의 보수 signed 정수의 최솟값과 최댓값을 반환한다."""
    _validate_bit_width(bit_width)
    half = 1 << (bit_width - 1)
    return -half, half - 1


def encode_unsigned(value: int, bit_width: int) -> str:
    """unsigned 정수를 고정 폭 이진 문자열로 인코딩한다."""
    _validate_bit_width(bit_width)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value는 정수여야 합니다.")

    maximum = state_count(bit_width) - 1
    if not 0 <= value <= maximum:
        raise ValueError(f"{bit_width}비트 unsigned 범위는 0~{maximum}입니다.")
    return format(value, f"0{bit_width}b")


def decode_unsigned(bits: str) -> int:
    """이진 문자열을 unsigned 정수로 해석한다."""
    _validate_bits(bits)
    return int(bits, 2)


def encode_twos_complement(value: int, bit_width: int) -> str:
    """signed 정수를 고정 폭 2의 보수 비트열로 인코딩한다."""
    _validate_bit_width(bit_width)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value는 정수여야 합니다.")

    minimum, maximum = signed_range(bit_width)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{bit_width}비트 signed 범위는 {minimum}~{maximum}입니다."
        )

    raw_code = value % state_count(bit_width)
    return format(raw_code, f"0{bit_width}b")


def decode_twos_complement(bits: str) -> int:
    """고정 폭 이진 문자열을 2의 보수 signed 정수로 해석한다."""
    _validate_bits(bits)
    raw_code = int(bits, 2)
    sign_bit = 1 << (len(bits) - 1)
    if raw_code & sign_bit:
        return raw_code - (1 << len(bits))
    return raw_code


def enumerate_interpretations(bit_width: int) -> tuple[IntegerInterpretation, ...]:
    """모든 비트열과 unsigned/signed 해석을 코드 순서대로 반환한다."""
    _validate_bit_width(bit_width)
    rows = []
    for raw_code in range(state_count(bit_width)):
        bits = encode_unsigned(raw_code, bit_width)
        rows.append(
            IntegerInterpretation(
                bits=bits,
                unsigned=decode_unsigned(bits),
                signed=decode_twos_complement(bits),
            )
        )
    return tuple(rows)
