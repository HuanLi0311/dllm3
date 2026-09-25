#!/usr/bin/env python3
"""Run the frozen Qwen3-1.7B version of the soft-target intervention."""

from pathlib import Path

from reproduction import ar_gsm8k_soft_targets as experiment
from reproduction.gsm8k_scale import MODEL_SPECS


ROOT = Path(__file__).resolve().parents[1]
SPEC = MODEL_SPECS["qwen3_1.7b"]
BASE_RUNNER = Path(experiment.__file__).resolve()

# ponytail: the validated 0.6B runner is model-agnostic; patch only its frozen
# provenance constants instead of maintaining a second copy of the experiment.
experiment.DEFAULT_MODEL = SPEC["path"]
experiment.MODEL_INVENTORY_SHA256 = SPEC["inventory_sha256"]
experiment.PROTOCOL = ROOT / "report/qwen17_gsm8k_soft_targets_protocol.md"
experiment.DEPENDENCIES = (*experiment.DEPENDENCIES, BASE_RUNNER)
experiment.__file__ = __file__


if __name__ == "__main__":
    experiment.main()
