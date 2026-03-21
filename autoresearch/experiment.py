"""Autoresearch experiment for hgx.

This file is modified by the autonomous research agent.
The agent changes parameters, models, and configurations here.
Infrastructure (imports, metric computation, output format) is fixed.
"""

import json
import sys
import time

# === EXPERIMENT CONFIGURATION (agent modifies this section) ===

EXPERIMENT = {
    "description": "baseline configuration",
    "parameters": {
        # Fill in project-specific parameters
    },
}

# === INFRASTRUCTURE (do not modify below this line) ===


def run_experiment(config: dict) -> dict:
    """Run a single experiment and return results."""
    start = time.time()

    # TODO: Import project-specific code and run experiment.
    # This is a placeholder — customize for hgx.
    result = {
        "metric_value": 0.0,
        "parameters": config["parameters"],
        "wall_time": time.time() - start,
        "status": "placeholder",
    }

    return result


def main():
    result = run_experiment(EXPERIMENT)

    # Output in parseable format.
    print(f"RESULT|{result['metric_value']}|{json.dumps(result['parameters'])}|{result['status']}|{EXPERIMENT['description']}")


if __name__ == "__main__":
    main()
