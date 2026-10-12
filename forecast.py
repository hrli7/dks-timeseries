"""Load TimesFM 3 from Hugging Face and forecast aligned CSV time series."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "google/timesfm-3.0-pytorch"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"


def read_csv(path, columns, freq):
    frame = pd.read_csv(path)
    required = ["date", *columns]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"CSV must include: {required}")
    frame = frame[required].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    dates = pd.DatetimeIndex(frame["date"])
    if (
        dates.empty
        or dates.hasnans
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
    ):
        raise ValueError("Dates must be nonempty, valid, unique and ascending.")
    expected = pd.date_range(dates[0], periods=len(dates), freq=freq)
    if not dates.equals(expected):
        raise ValueError(
            "Dates must follow --freq exactly; resolve gaps before forecasting."
        )
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[columns].to_numpy(dtype=np.float32)).all():
        raise ValueError(
            "Selected columns must contain finite numeric values without gaps."
        )
    return frame


def prepare(history, targets, past, known, horizon, freq, future=None, holdout=False):
    columns = targets + past + known
    if not targets or len(columns) != len(set(columns)) or "date" in columns:
        raise ValueError(
            "Choose targets and disjoint covariate columns; date is reserved."
        )
    if horizon < 1:
        raise ValueError("Horizon must be positive.")
    frame = read_csv(history, columns, freq)
    if holdout and future:
        raise ValueError(
            "Holdout uses the final historical rows; do not supply --future."
        )
    train = frame.iloc[:-horizon].copy() if holdout else frame
    if not 32 <= len(train) <= 15360:
        raise ValueError(
            "Supply 32–15360 training observations; trim a longer history explicitly."
        )
    dates = pd.date_range(train.date.iloc[-1], periods=horizon + 1, freq=freq)[1:]
    actual = frame.iloc[-horizon:].copy() if holdout else None
    future_frame = actual
    if known and not holdout:
        if future is None:
            raise ValueError(
                "--known-covariates requires --future with the full forecast horizon."
            )
        future_frame = read_csv(future, known, freq)
        if not pd.DatetimeIndex(future_frame.date).equals(dates):
            raise ValueError(
                "Future dates must exactly cover the next horizon after history."
            )
    elif future and not known:
        raise ValueError("--future requires --known-covariates.")

    def matrix(data, names):
        return data[names].to_numpy(dtype=np.float32).T if names else None

    return (
        {
            "context": matrix(train, targets),
            "past_only_covariates": matrix(train, past),
            "past_future_covariates": (
                np.concatenate(
                    [matrix(train, known), matrix(future_frame, known)], axis=1
                )
                if known
                else None
            ),
        },
        dates,
        train,
        actual,
    )


def forecast_frame(output, dates, targets):
    points = np.asarray(output.forecast)
    quantiles = np.asarray(output.quantiles)
    shape = (len(targets), len(dates))
    if points.shape != shape or quantiles.shape != (*shape, 9):
        raise ValueError("Unexpected model output dimensions.")
    if not np.isfinite(points).all() or not np.isfinite(quantiles).all():
        raise ValueError("Model returned nonfinite forecasts.")
    rows = []
    for i, name in enumerate(targets):
        for t, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "series": name,
                    "forecast": float(points[i, t]),
                    **{
                        f"p{q * 10}": float(quantiles[i, t, q - 1])
                        for q in range(1, 10)
                    },
                }
            )
    return pd.DataFrame(rows)


def score(predictions, actual, train, targets):
    scores = {}
    for name in targets:
        y = actual[name].to_numpy(dtype=float)
        pred = predictions.loc[predictions.series == name, "forecast"].to_numpy()
        error = pred - y
        denominator = float(np.abs(y).sum())
        scores[name] = {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "wape": float(np.abs(error).sum() / denominator) if denominator else None,
            "last_value_mae": float(np.abs(y - train[name].iloc[-1]).mean()),
        }
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--past-covariates", nargs="+", default=[])
    p.add_argument("--known-covariates", nargs="+", default=[])
    p.add_argument("--future", type=Path)
    p.add_argument(
        "--freq", required=True, help="Pandas frequency, e.g. W-MON, MS, or D"
    )
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--holdout", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    arrays, dates, train, actual = prepare(
        args.input,
        args.targets,
        args.past_covariates,
        args.known_covariates,
        args.horizon,
        args.freq,
        args.future,
        args.holdout,
    )
    if args.validate_only:
        print(
            "Input valid:",
            {k: None if v is None else list(v.shape) for k, v in arrays.items()},
        )
        return
    if args.output.exists():
        raise FileExistsError("Output exists; choose a new run directory.")
    from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster

    model = TimesFM3Forecaster.from_pretrained(
        MODEL,
        revision=args.revision,
        device=args.device,
        per_core_batch_size=1,
    )
    output = model.predict(**arrays, horizon=args.horizon, return_quantiles=True)
    predictions = forecast_frame(output, dates, args.targets)
    args.output.mkdir(parents=True)
    predictions.to_csv(args.output / "forecast.csv", index=False)
    if actual is not None:
        (args.output / "metrics.json").write_text(
            json.dumps(
                score(predictions, actual, train, args.targets),
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    manifest = {
        "model": MODEL,
        "revision": args.revision,
        "device": args.device,
        "targets": args.targets,
        "past_covariates": args.past_covariates,
        "known_covariates": args.known_covariates,
        "frequency": args.freq,
        "horizon": args.horizon,
        "holdout": args.holdout,
        "training_rows": len(train),
        "training_end": str(train.date.iloc[-1]),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "future_sha256": hashlib.sha256(args.future.read_bytes()).hexdigest()
        if args.future
        else None,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["timesfm", "torch", "numpy", "pandas"]
        },
        "point_forecast": "median; quantiles are marginal per series and date",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved forecast and provenance to {args.output}")


if __name__ == "__main__":
    main()
