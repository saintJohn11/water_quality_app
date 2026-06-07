"""
Pure numpy inference for the Deep Learning model.
No TensorFlow or Keras needed — just numpy!
Drop this file in your repo alongside dl_model_weights.json
"""
import numpy as np
import json
import os

def _relu(x): return np.maximum(0, x)
def _sigmoid(x): return 1 / (1 + np.exp(-x))

def load_numpy_model(weights_path="dl_model_weights.json"):
    with open(weights_path, "r") as f:
        return json.load(f)

def numpy_predict(layers_data, X):
    """Run inference. X shape: (n_samples, 9). Returns probabilities (n_samples,)."""
    out = np.array(X, dtype=np.float64)
    for layer in layers_data:
        t = layer["type"]
        w = layer["weights"]
        if t == "Dense":
            W, b = np.array(w[0]), np.array(w[1])
            out = out @ W + b
            out = _sigmoid(out) if W.shape[1] == 1 else _relu(out)
        elif t == "BatchNormalization":
            gamma, beta, mean, var = [np.array(x) for x in w]
            out = gamma * (out - mean) / np.sqrt(var + 1e-3) + beta
        # Dropout is a no-op at inference time
    return out.flatten()
