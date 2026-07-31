"""4비트의 모든 상태를 unsigned와 signed로 비교한다."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bits_and_integers import (
    decode_twos_complement,
    decode_unsigned,
    encode_twos_complement,
    enumerate_interpretations,
    signed_range,
    state_count,
)


def main() -> None:
    bit_width = 4
    minimum, maximum = signed_range(bit_width)

    print(f"{bit_width}비트 상태 수: {state_count(bit_width)}")
    print(f"unsigned 범위: 0~{state_count(bit_width) - 1}")
    print(f"signed 범위: {minimum}~{maximum}")
    print()
    print("bits  unsigned  signed")
    print("----  --------  ------")

    for row in enumerate_interpretations(bit_width):
        print(f"{row.bits}  {row.unsigned:8d}  {row.signed:6d}")

    sample_bits = "1101"
    print("\n같은 비트열의 다른 해석")
    print(f"{sample_bits} -> unsigned {decode_unsigned(sample_bits)}")
    print(f"{sample_bits} -> signed {decode_twos_complement(sample_bits)}")

    value = -3
    print("\n2의 보수 인코딩")
    print(f"{value} -> {encode_twos_complement(value, bit_width)}")


if __name__ == "__main__":
    main()
