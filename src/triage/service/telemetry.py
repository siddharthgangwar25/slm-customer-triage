"""Bounded process-local Prometheus metrics; never label with request content or IDs."""

import json
import threading
from collections import Counter


class Telemetry:
    buckets = (0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 120)

    def __init__(self):
        self.lock = threading.Lock()
        self.counts = Counter()
        self.sums = Counter()
        self.histograms = {}
        self.inflight = 0
        self.model_info = None

    def increment(self, name, value=1):
        with self.lock:
            self.counts[name] += value

    def observe(self, name, seconds):
        with self.lock:
            self.sums[name] += seconds
            bins = self.histograms.setdefault(name, [0] * (len(self.buckets) + 1))
            for i, bound in enumerate(self.buckets):
                bins[i] += seconds <= bound
            bins[-1] += 1

    def active(self, change):
        with self.lock:
            self.inflight += change

    def render(self):
        with self.lock:
            lines = ["# TYPE triage_inflight gauge", f"triage_inflight {self.inflight}"]
            if self.model_info:
                model, policy = map(json.dumps, self.model_info)
                lines.append(
                    f"triage_model_info{{model_version={model},policy_version={policy}}} 1"
                )
            for name, value in sorted(self.counts.items()):
                lines += [f"# TYPE triage_{name}_total counter", f"triage_{name}_total {value}"]
            for name, bins in sorted(self.histograms.items()):
                lines.append(f"# TYPE triage_{name}_seconds histogram")
                for bound, value in zip((*self.buckets, "+Inf"), bins, strict=True):
                    lines.append(f'triage_{name}_seconds_bucket{{le="{bound}"}} {value}')
                lines += [
                    f"triage_{name}_seconds_count {bins[-1]}",
                    f"triage_{name}_seconds_sum {self.sums[name]}",
                ]
            return "\n".join(lines) + "\n"
