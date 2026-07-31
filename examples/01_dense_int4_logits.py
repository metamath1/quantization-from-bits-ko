"""작은 Dense 모델에서 INT4 양자화 전후의 로짓 순위를 비교한다.

핵심 메시지:
1. 역양자화된 가중치는 원본과 같지 않다.
2. 따라서 로짓도 달라진다.
3. 하지만 클래스 간 마진이 충분하면 top-1 순위는 대부분 유지된다.
4. 결정경계 근처에서는 작은 오차로 순위가 바뀔 수 있다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.int4_quant import quantize_dequantize_int4_per_row


RNG = np.random.default_rng(7)


@dataclass
class DenseModel:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def logits(self, x: np.ndarray) -> np.ndarray:
        hidden = np.tanh(x @ self.w1.T + self.b1)
        return hidden @ self.w2.T + self.b2


def make_dataset(n_per_class: int = 250) -> tuple[np.ndarray, np.ndarray]:
    centers = np.array(
        [
            [-2.0, -1.4],
            [2.1, -1.1],
            [0.1, 2.2],
        ],
        dtype=np.float32,
    )
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for class_id, center in enumerate(centers):
        points = RNG.normal(loc=center, scale=0.55, size=(n_per_class, 2))
        xs.append(points.astype(np.float32))
        ys.append(np.full(n_per_class, class_id, dtype=np.int64))

    x = np.vstack(xs)
    y = np.concatenate(ys)
    order = RNG.permutation(len(x))
    return x[order], y[order]


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    result = np.zeros((len(y), n_classes), dtype=np.float32)
    result[np.arange(len(y)), y] = 1.0
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def train_model(x: np.ndarray, y: np.ndarray) -> DenseModel:
    hidden_size = 12
    n_classes = 3

    model = DenseModel(
        w1=RNG.normal(0.0, 0.45, size=(hidden_size, 2)).astype(np.float32),
        b1=np.zeros(hidden_size, dtype=np.float32),
        w2=RNG.normal(0.0, 0.35, size=(n_classes, hidden_size)).astype(np.float32),
        b2=np.zeros(n_classes, dtype=np.float32),
    )

    target = one_hot(y, n_classes)
    learning_rate = 0.06

    for _ in range(900):
        z1 = x @ model.w1.T + model.b1
        h = np.tanh(z1)
        logits = h @ model.w2.T + model.b2
        probs = softmax(logits)

        d_logits = (probs - target) / len(x)
        d_w2 = d_logits.T @ h
        d_b2 = np.sum(d_logits, axis=0)

        d_h = d_logits @ model.w2
        d_z1 = d_h * (1.0 - h**2)
        d_w1 = d_z1.T @ x
        d_b1 = np.sum(d_z1, axis=0)

        model.w1 -= learning_rate * d_w1
        model.b1 -= learning_rate * d_b1
        model.w2 -= learning_rate * d_w2
        model.b2 -= learning_rate * d_b2

    return model


def quantize_model(model: DenseModel) -> DenseModel:
    return DenseModel(
        w1=quantize_dequantize_int4_per_row(model.w1),
        b1=model.b1.copy(),
        w2=quantize_dequantize_int4_per_row(model.w2),
        b2=model.b2.copy(),
    )


def ranking(logits: np.ndarray) -> np.ndarray:
    return np.argsort(-logits, axis=1)


def top2_margin(logits: np.ndarray) -> np.ndarray:
    sorted_logits = np.sort(logits, axis=1)
    return sorted_logits[:, -1] - sorted_logits[:, -2]


def print_sample(
    title: str,
    x: np.ndarray,
    fp_logits: np.ndarray,
    int4_logits: np.ndarray,
) -> None:
    print(f"\n[{title}]")
    print("입력:", np.array2string(x, precision=4))
    print("FP32 logits:", np.array2string(fp_logits, precision=5))
    print("INT4 logits:", np.array2string(int4_logits, precision=5))
    print("FP32 순위:", np.argsort(-fp_logits).tolist())
    print("INT4 순위:", np.argsort(-int4_logits).tolist())


def main() -> None:
    x, y = make_dataset()
    split = int(len(x) * 0.75)
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    fp_model = train_model(x_train, y_train)
    int4_model = quantize_model(fp_model)

    fp_logits = fp_model.logits(x_test)
    int4_logits = int4_model.logits(x_test)

    fp_pred = np.argmax(fp_logits, axis=1)
    int4_pred = np.argmax(int4_logits, axis=1)

    fp_rank = ranking(fp_logits)
    int4_rank = ranking(int4_logits)
    margins = top2_margin(fp_logits)

    weight_mse = (
        np.mean((fp_model.w1 - int4_model.w1) ** 2)
        + np.mean((fp_model.w2 - int4_model.w2) ** 2)
    ) / 2.0
    logit_mae = np.mean(np.abs(fp_logits - int4_logits))
    logit_max_error = np.max(np.abs(fp_logits - int4_logits))
    top1_agreement = np.mean(fp_pred == int4_pred)
    full_rank_agreement = np.mean(np.all(fp_rank == int4_rank, axis=1))
    fp_accuracy = np.mean(fp_pred == y_test)
    int4_accuracy = np.mean(int4_pred == y_test)

    print("=== Dense toy model: FP32 vs symmetric INT4 weight-only ===")
    print(f"테스트 샘플 수: {len(x_test)}")
    print(f"가중치 MSE: {weight_mse:.8f}")
    print(f"로짓 평균 절대오차: {logit_mae:.6f}")
    print(f"로짓 최대 절대오차: {logit_max_error:.6f}")
    print(f"FP32 정확도: {fp_accuracy:.2%}")
    print(f"INT4 정확도: {int4_accuracy:.2%}")
    print(f"top-1 예측 일치율: {top1_agreement:.2%}")
    print(f"전체 로짓 순위 일치율: {full_rank_agreement:.2%}")

    high_margin = margins >= np.quantile(margins, 0.75)
    low_margin = margins <= np.quantile(margins, 0.10)
    print(f"상위 25% 마진 샘플 top-1 일치율: {np.mean(fp_pred[high_margin] == int4_pred[high_margin]):.2%}")
    print(f"하위 10% 마진 샘플 top-1 일치율: {np.mean(fp_pred[low_margin] == int4_pred[low_margin]):.2%}")

    stable_candidates = np.where(np.all(fp_rank == int4_rank, axis=1))[0]
    stable_idx = stable_candidates[np.argmax(margins[stable_candidates])]
    print_sample(
        "마진이 큰 안정적인 샘플",
        x_test[stable_idx],
        fp_logits[stable_idx],
        int4_logits[stable_idx],
    )

    changed = np.where(np.any(fp_rank != int4_rank, axis=1))[0]
    if len(changed) > 0:
        changed_idx = changed[np.argmin(margins[changed])]
        print_sample(
            "결정경계 부근의 순위 변화 샘플",
            x_test[changed_idx],
            fp_logits[changed_idx],
            int4_logits[changed_idx],
        )
    else:
        boundary_idx = int(np.argmin(margins))
        print_sample(
            "가장 작은 마진의 샘플(이번 실행에서는 순위 유지)",
            x_test[boundary_idx],
            fp_logits[boundary_idx],
            int4_logits[boundary_idx],
        )

    print("\n해석:")
    print("- 역양자화된 가중치와 로짓은 원본과 다릅니다.")
    print("- 그러나 클래스 간 로짓 마진이 양자화 오차보다 크면 top-1 순위는 유지됩니다.")
    print("- 결정경계 근처처럼 마진이 작은 입력은 작은 오차에도 순위가 바뀔 수 있습니다.")


if __name__ == "__main__":
    main()
