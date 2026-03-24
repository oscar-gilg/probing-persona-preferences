"""Check if original A and B tasks are systematically different."""

import json
import numpy as np
from pathlib import Path

with open("experiments/steering/cross_layer/pairs_500.json") as f:
    pairs = json.load(f)

mu_a = [p["mu_a"] for p in pairs]
mu_b = [p["mu_b"] for p in pairs]
delta = [p["delta_mu"] for p in pairs]

print(f"mean mu_a: {np.mean(mu_a):.3f}")
print(f"mean mu_b: {np.mean(mu_b):.3f}")
print(f"mean delta_mu (A-B): {np.mean(delta):.3f}")
print(f"P(mu_a > mu_b): {np.mean([a > b for a, b in zip(mu_a, mu_b)]):.3f}")
print(f"std delta_mu: {np.std(delta):.3f}")
