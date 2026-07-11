"""Visualize breathing data stored in a NumPy .npy file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
	parser = argparse.ArgumentParser(description="Visualize breathe_data.npy")
	parser.add_argument(
		"path",
		nargs="?",
		default="breathe_data.npy",
		help="Path to the breathe_data.npy file",
	)
	args = parser.parse_args()

	path = Path(args.path)
	data = np.load(path)

	plt.figure(figsize=(10, 4))
	if data.ndim == 1:
		plt.plot(data, label="signal", linewidth=10)
	else:
		for idx in range(data.shape[-1]):
			plt.plot(data[..., idx], label=f"channel {idx}", linewidth=10)

	plt.title("Breathing Data")
	plt.xlabel("Sample")
	plt.ylabel("Value")
	plt.grid(True, alpha=0.3)
	if data.ndim != 1:
		plt.legend()
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
