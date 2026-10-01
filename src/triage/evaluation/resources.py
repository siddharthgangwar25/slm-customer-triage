"""Optional nvidia-smi samples. Absence is explicit, never inferred as zero utilization."""

import os
import subprocess
import threading
import time


class GPUProbe:
    def __init__(self):
        self.samples = []
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                values = [float(v.strip()) for v in result.stdout.splitlines()[0].split(",")]
                self.samples.append(
                    {
                        "unix_seconds": time.time(),
                        "utilization_percent": values[0],
                        "memory_used_mib": values[1],
                        "memory_total_mib": values[2],
                    }
                )
            except (OSError, ValueError, subprocess.SubprocessError):
                self.error = "nvidia-smi unavailable or unsupported"
                return
            self.stop.wait(2)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=6)
        return {
            "samples": self.samples,
            "error": self.error,
            "scope": "whole GPU, includes unrelated desktop processes; 2-second sampling",
        }
