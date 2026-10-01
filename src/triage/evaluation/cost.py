"""Explicit dated price scenarios; local throughput is never called a measured cloud result."""

import math
from pathlib import Path

from triage.io import read_json, write_json


def estimate(
    load, price, *, monthly_requests, monthly_hours, supporting_monthly_usd, training_hours
):
    values = [price, monthly_requests, monthly_hours, supporting_monthly_usd, training_hours]
    if any(not math.isfinite(v) or v < 0 for v in values) or monthly_requests == 0:
        raise ValueError("Cost assumptions must be finite/nonnegative with positive demand")
    serial = next(r for r in load["runs"] if r["concurrency"] == 1)
    active_seconds = serial["elapsed_seconds"] / serial["submitted"]
    active_hours = monthly_requests * active_seconds / 3600
    hosting = price * monthly_hours + supporting_monthly_usd
    return {
        "monthly_submitted_requests_assumed": monthly_requests,
        "billed_monthly_hours_assumed": monthly_hours,
        "supporting_monthly_usd_assumed": supporting_monthly_usd,
        "active_hours_at_local_measured_rate": active_hours,
        "idle_hours_at_local_measured_rate": max(0, monthly_hours - active_hours),
        "demand_fits_assumed_capacity": active_hours <= monthly_hours,
        "hosting_usd_per_month": hosting,
        "hosting_usd_per_1000_submitted": hosting / monthly_requests * 1000,
        "active_compute_usd_per_1000_submitted": price * active_seconds / 3600 * 1000,
        "training_hours_local": training_hours,
        "training_usd_at_scenario_rate": training_hours * price,
        "training_amortization_one_month_usd_per_1000": training_hours
        * price
        / monthly_requests
        * 1000,
        "measured_cloud_cost": False,
        "limitation": "Cloud runtime is unmeasured. Local GTX 1650 runtime at a T4 host price "
        "is an explicit what-if, not a cloud performance/cost claim. Supporting allowance is "
        "assumed; excludes taxes. No spending occurred.",
    }


def report(load_path, output):
    quotes = read_json(Path("deployment/pricing/ec2-us-east-1-2026-09-29.json"))
    hourly = float(
        next(r["price"] for r in quotes["quotes"] if r["Instance Type"] == "g4dn.xlarge")
    )
    load = read_json(load_path)
    training = read_json(Path("reports/finetuning-run-v1/result.json"))
    result = {
        "quote": quotes,
        "hardware_price_scenario": "g4dn.xlarge Linux On-Demand, us-east-1",
        "measured_hardware": "local GTX 1650 + CPU gateway",
        "load_complete": load["complete"],
        "scenario": estimate(
            load,
            hourly,
            monthly_requests=100000,
            monthly_hours=730,
            supporting_monthly_usd=10,
            training_hours=training["training_seconds_this_session"] / 3600,
        ),
    }
    write_json(output, result)
    return result
