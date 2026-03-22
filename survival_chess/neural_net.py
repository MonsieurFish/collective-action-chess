"""
MLP neural network operations for survival chess.
No autograd needed — neuroevolution uses direct weight mutation.

Network architecture:
  Input (45) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(N)

Each network is stored as a plain dict of numpy arrays:
  {
    'W1': (input_size, 64),
    'b1': (64,),
    'W2': (64, 32),
    'b2': (32,),
    'W3': (32, output_size),
    'b3': (output_size,),
  }

All weights initialized from Uniform(-0.5, 0.5).
"""

import numpy as np
from typing import Dict

Weights = Dict[str, np.ndarray]


def make_network(input_size: int, hidden1: int, hidden2: int, output_size: int) -> Weights:
    """Initialize a new MLP with weights drawn from Uniform(-0.5, 0.5)."""
    return {
        "W1": np.random.uniform(-0.5, 0.5, (input_size, hidden1)).astype(np.float32),
        "b1": np.random.uniform(-0.5, 0.5, (hidden1,)).astype(np.float32),
        "W2": np.random.uniform(-0.5, 0.5, (hidden1, hidden2)).astype(np.float32),
        "b2": np.random.uniform(-0.5, 0.5, (hidden2,)).astype(np.float32),
        "W3": np.random.uniform(-0.5, 0.5, (hidden2, output_size)).astype(np.float32),
        "b3": np.random.uniform(-0.5, 0.5, (output_size,)).astype(np.float32),
    }


def forward(weights: Weights, x: np.ndarray) -> np.ndarray:
    """Forward pass. Returns raw logits (no activation on output layer)."""
    h1 = np.maximum(0.0, x @ weights["W1"] + weights["b1"])
    h2 = np.maximum(0.0, h1 @ weights["W2"] + weights["b2"])
    return h2 @ weights["W3"] + weights["b3"]


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Numerically stable softmax with temperature.
    Entries that are -inf (masked) get probability 0.
    """
    logits = np.array(logits, dtype=np.float64)
    logits = logits / max(temperature, 1e-8)

    finite_mask = np.isfinite(logits)
    if not np.any(finite_mask):
        # Degenerate: all masked — return uniform (should not happen)
        n = len(logits)
        return np.full(n, 1.0 / n)

    # Subtract max of finite values for numerical stability
    max_val = np.max(logits[finite_mask])
    shifted = logits - max_val
    shifted = np.clip(shifted, -500, 0)  # prevent overflow; -inf stays -inf

    exp_vals = np.exp(shifted)
    exp_vals[~finite_mask] = 0.0  # masked entries → 0 probability

    total = exp_vals.sum()
    if total == 0.0:
        # Fallback: uniform over unmasked entries
        probs = np.zeros(len(logits))
        probs[finite_mask] = 1.0 / np.sum(finite_mask)
        return probs

    return (exp_vals / total).astype(np.float64)


def mutate(weights: Weights, mutation_rate: float, mutation_std: float) -> Weights:
    """
    Return a new weight dict where each weight, independently with probability
    `mutation_rate`, has Gaussian noise N(0, mutation_std) added.
    """
    new_weights = {}
    for key, w in weights.items():
        mask = np.random.random(w.shape) < mutation_rate
        noise = np.random.normal(0.0, mutation_std, w.shape).astype(w.dtype)
        new_weights[key] = w + mask * noise
    return new_weights


def copy_weights(weights: Weights) -> Weights:
    """Deep copy a weight dict."""
    return {k: v.copy() for k, v in weights.items()}
