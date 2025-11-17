"""Utilities for generating synthetic data for customer purchase prediction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.datasets import make_classification


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "customer_data.csv"


def generate_data(
    n_samples: int = 2000,
    random_state: int = 42,
    class_sep: float = 1.2,
    weights: Sequence[float] | None = (0.6, 0.4),
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate a synthetic customer purchase dataset and persist it to CSV.

    Parameters
    ----------
    n_samples:
        Total number of rows to generate. Defaults to ``2000``.
    random_state:
        Seed for deterministic data generation. Defaults to ``42``.
    class_sep:
        Separation between the generated classes. Defaults to ``1.2``.
    weights:
        Class balance weights passed to ``make_classification``. Defaults to ``(0.6, 0.4)``.
    output_path:
        Target CSV path. Defaults to ``data/customer_data.csv``.

    Returns
    -------
    pandas.DataFrame
        The generated dataset containing feature columns ``f1``..``f10`` and ``target``.
    """

    output_path = output_path or DEFAULT_OUTPUT_PATH
    features, target = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=list(weights) if weights is not None else None,
        class_sep=class_sep,
        random_state=random_state,
    )

    column_names = [f"f{idx}" for idx in range(1, 11)]
    df = pd.DataFrame(features, columns=column_names)
    df["target"] = target

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset generation."""

    parser = argparse.ArgumentParser(
        description="Generate synthetic data for the customer purchase prediction task.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Total number of samples to generate (defaults to 2000 or DATA_N_SAMPLES).",
    )
    parser.add_argument(
        "--class-sep",
        type=float,
        default=None,
        help="Class separation factor (defaults to 1.2 or DATA_CLASS_SEP).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination CSV file (defaults to data/customer_data.csv).",
    )
    return parser.parse_args()


def _resolve_parameter(env_key: str, cli_value: int | float | None, default: int | float) -> int | float:
    """Resolve parameter precedence: CLI overrides env, which overrides default."""

    if cli_value is not None:
        return cli_value

    env_value = os.getenv(env_key)
    if env_value is not None:
        try:
            return type(default)(env_value)
        except ValueError as exc:  # pragma: no cover - defensive programming
            raise ValueError(f"Environment variable {env_key} must be a {type(default).__name__}.") from exc

    return default


if __name__ == "__main__":
    print("Starting data generation...")
    args = _parse_args()
    n_samples = _resolve_parameter("DATA_N_SAMPLES", args.n_samples, 2000)
    class_sep = _resolve_parameter("DATA_CLASS_SEP", args.class_sep, 1.2)

    dataset = generate_data(
        n_samples=n_samples,
        class_sep=class_sep,
        random_state=42,
        weights=(0.6, 0.4),
        output_path=args.output_path,
    )
    print(f"Generated dataset with shape {dataset.shape} at {args.output_path}")

