"""HTTP layer.

The engine knows nothing about the web: it takes price frames and returns
objects. Everything to do with parsing a request, turning pandas objects into
JSON, and reporting failure with a status code lives here.

Serialisation is deliberately explicit rather than automatic. NaN is not valid
JSON and numpy scalars are not JSON-serialisable, so every number that leaves
this module is passed through `_number` first.
"""

import argparse
import math

from flask import Flask, jsonify, render_template, request

from portfolio.analysis import analyse
from portfolio.data import DataError, fetch_prices
from portfolio.simulation import SimulationError, simulate

DEFAULT_BENCHMARK = "^GSPC"
REBALANCE_CHOICES = ("none", "monthly", "quarterly", "annual")


class RequestError(Exception):
    """Raised when the submitted form cannot be turned into a simulation."""


# --------------------------- request parsing ---------------------------

def _number(value, default=None):
    """Coerce to a JSON-safe float. NaN and infinity become `default`."""
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _required_float(payload, key, minimum=None, allow_zero=True):
    raw = payload.get(key)
    if raw is None or raw == "":
        raise RequestError(f"{key.replace('_', ' ').capitalize()} is required.")
    value = _number(raw)
    if value is None:
        raise RequestError(f"{key.replace('_', ' ').capitalize()} must be a number.")
    if minimum is not None and (value < minimum or (not allow_zero and value == 0)):
        raise RequestError(
            f"{key.replace('_', ' ').capitalize()} must be greater than {minimum}."
        )
    return value


def parse_request(payload):
    """Turn a submitted JSON body into validated simulation arguments.

    Everything user-supplied is checked here so that a bad form produces a 400
    with a readable message rather than a 500 from somewhere in pandas.
    """
    tickers = payload.get("tickers") or []
    if isinstance(tickers, str):
        tickers = tickers.replace(",", " ").split()
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        raise RequestError("Enter at least one ticker.")
    if len(tickers) > 10:
        raise RequestError("Ten tickers is the maximum.")

    weights = payload.get("weights") or [1.0] * len(tickers)
    if len(weights) != len(tickers):
        raise RequestError("Each ticker needs a weight.")
    weights = [_number(w, 0.0) for w in weights]
    if any(w < 0 for w in weights):
        raise RequestError("Weights cannot be negative.")
    if sum(weights) <= 0:
        raise RequestError("Weights must add up to more than zero.")

    start_date = str(payload.get("start_date") or "").strip()
    end_date = str(payload.get("end_date") or "").strip()
    if not start_date or not end_date:
        raise RequestError("Both a start and an end date are required.")
    if start_date >= end_date:
        raise RequestError("The start date must come before the end date.")

    rebalance = str(payload.get("rebalance") or "none").lower()
    if rebalance not in REBALANCE_CHOICES:
        raise RequestError(f"Rebalancing must be one of: {', '.join(REBALANCE_CHOICES)}.")

    benchmark = str(payload.get("benchmark") or DEFAULT_BENCHMARK).strip().upper()

    return {
        "tickers": tickers,
        "weights": weights,
        "start_date": start_date,
        "end_date": end_date,
        "initial_investment": _required_float(
            payload, "initial_investment", minimum=0, allow_zero=False
        ),
        "monthly_contribution": _number(payload.get("monthly_contribution"), 0.0),
        "rebalance": rebalance,
        "benchmark": benchmark or None,
        "risk_free_rate": _number(payload.get("risk_free_rate"), 0.0) / 100,
    }


# --------------------------- serialisation ---------------------------

def _dates(index):
    return [d.strftime("%Y-%m-%d") for d in index]


def _values(series):
    return [_number(v) for v in series.tolist()]


def _summary(stats):
    """Every scalar the front end shows, made JSON-safe in one place."""
    return {key: (_number(value) if isinstance(value, (int, float)) else value)
            for key, value in stats.items()}


def build_response(result, stats, benchmark_result, benchmark_stats,
                   benchmark_ticker, correlation, annual, drawdown):
    allocation = []
    final_prices = result.prices.iloc[-1]
    final_holdings = result.shares.iloc[-1] * final_prices
    final_total = final_holdings.sum()

    for ticker in result.prices.columns:
        allocation.append({
            "ticker": ticker,
            "target": _number(result.weights[ticker]),
            "final": _number(final_holdings[ticker] / final_total if final_total else 0),
            "value": _number(final_holdings[ticker]),
        })

    payload = {
        "tickers": list(result.prices.columns),
        "start": _dates(result.prices.index[:1])[0],
        "end": _dates(result.prices.index[-1:])[0],
        "summary": _summary(stats),
        "allocation": allocation,
        "series": {
            "dates": _dates(result.values.index),
            "portfolio": _values(result.values),
            "invested": _values(result.invested),
            "drawdown": _values(drawdown) if drawdown is not None else None,
        },
        "correlation": {
            "tickers": list(correlation.columns),
            "matrix": [[_number(v) for v in row] for row in correlation.to_numpy()],
        } if correlation is not None else None,
        "annual_returns": annual,
        "benchmark": None,
    }

    if benchmark_result is not None:
        payload["benchmark"] = {
            "ticker": benchmark_ticker,
            "summary": _summary(benchmark_stats),
            "values": _values(benchmark_result.values),
        }

    return payload


# --------------------------- app ---------------------------

def create_app():
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/simulate")
    def run_simulation():
        payload = request.get_json(silent=True) or {}

        try:
            args = parse_request(payload)
        except RequestError as exc:
            return jsonify(error=str(exc)), 400

        try:
            prices = fetch_prices(args["tickers"], args["start_date"], args["end_date"])
        except DataError as exc:
            return jsonify(error=str(exc)), 400

        benchmark_prices = None
        if args["benchmark"] and args["benchmark"] not in args["tickers"]:
            try:
                benchmark_prices = fetch_prices(
                    [args["benchmark"]], args["start_date"], args["end_date"]
                ).reindex(prices.index).ffill().dropna()
            except DataError:
                benchmark_prices = None  # a missing benchmark is not fatal

        try:
            result = simulate(
                prices,
                args["weights"],
                args["initial_investment"],
                monthly_contribution=args["monthly_contribution"],
                rebalance=args["rebalance"],
            )

            benchmark_result = None
            if benchmark_prices is not None and len(benchmark_prices) >= 2:
                benchmark_result = simulate(
                    benchmark_prices,
                    [1.0],
                    args["initial_investment"],
                    monthly_contribution=args["monthly_contribution"],
                )

            bundle = analyse(
                result,
                benchmark=benchmark_result,
                risk_free_rate=args["risk_free_rate"],
            )
        except SimulationError as exc:
            return jsonify(error=str(exc)), 400

        return jsonify(build_response(
            result,
            bundle["summary"],
            benchmark_result,
            bundle.get("benchmark_summary"),
            args["benchmark"],
            bundle.get("correlation"),
            bundle.get("annual_returns"),
            bundle.get("drawdown"),
        ))

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    return app


def main():
    parser = argparse.ArgumentParser(description="Portfolio simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-debug", action="store_true")
    options = parser.parse_args()

    create_app().run(host=options.host, port=options.port,
                     debug=not options.no_debug)


if __name__ == "__main__":
    main()
