"""교육용 대칭형 INT4 weight-only 양자화 함수.

성능 최적화가 아니라 원리를 명확하게 보여주는 것이 목적이다.
실제 GPTQ/AWQ/Marlin 구현을 대체하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantizedInt4:
    """행 단위 대칭형 INT4 양자화 결과."""

    codes: np.ndarray
    scales: np.ndarray


def quantize_symmetric_int4_per_row(weights: np.ndarray) -> QuantizedInt4:
    """2차원 가중치 행렬을 행별로 signed INT4(-7~7) 양자화한다.

    각 출력 뉴런의 가중치 행마다 별도 scale을 사용한다. -8을 사용하지 않고
    -7~7의 대칭 범위를 사용해 0을 정확히 표현한다.
    """
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"2차원 행렬이 필요합니다. 입력 shape={w.shape}")

    max_abs = np.max(np.abs(w), axis=1, keepdims=True)
    scales = np.where(max_abs == 0.0, 1.0, max_abs / 7.0).astype(np.float32)
    codes = np.clip(np.rint(w / scales), -7, 7).astype(np.int8)
    return QuantizedInt4(codes=codes, scales=scales)


def dequantize_int4(q: QuantizedInt4) -> np.ndarray:
    """INT4 코드와 scale을 근사 FP32 가중치로 해석한다."""
    return q.codes.astype(np.float32) * q.scales


def quantize_dequantize_int4_per_row(weights: np.ndarray) -> np.ndarray:
    """양자화 후 즉시 역양자화한 교육용 근사 가중치를 반환한다."""
    return dequantize_int4(quantize_symmetric_int4_per_row(weights))
