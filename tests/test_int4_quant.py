import numpy as np

from src.int4_quant import (
    dequantize_int4,
    quantize_symmetric_int4_per_row,
)


def test_codes_are_in_int4_range() -> None:
    weights = np.array([[1.0, -0.5, 0.25], [-2.0, 0.0, 1.0]], dtype=np.float32)
    quantized = quantize_symmetric_int4_per_row(weights)
    assert quantized.codes.min() >= -7
    assert quantized.codes.max() <= 7


def test_zero_row_is_stable() -> None:
    weights = np.zeros((2, 4), dtype=np.float32)
    restored = dequantize_int4(quantize_symmetric_int4_per_row(weights))
    np.testing.assert_array_equal(restored, weights)


def test_shape_is_preserved() -> None:
    weights = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
    restored = dequantize_int4(quantize_symmetric_int4_per_row(weights))
    assert restored.shape == weights.shape
