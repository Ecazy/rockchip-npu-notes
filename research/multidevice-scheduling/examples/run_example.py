#!/usr/bin/env python3
"""Run the two-device example scenario under all four scheduling policies.

Prints one compact metrics row per policy.  RK182X is a hypothetical device in
the example; no RK182X hardware claim is made.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from simulator import POLICIES, load_scenario, simulate  # noqa: E402

SCENARIO = HERE / "two_device_scenario.json"


def main():
    scenario = load_scenario(SCENARIO)
    print(f"scenario: {scenario.meta.get('name')}")
    print(f"{'policy':<16}{'makespan_ns':>13}{'rk3588_jobs':>13}{'rk182x_jobs':>13}{'transfer_bytes':>16}")
    for policy in POLICIES:
        result = simulate(scenario, policy)
        jobs = [j for j in result["jobs"] if j.get("state", "done") == "done"]
        rk3588 = sum(j["device"] == "rk3588" for j in jobs)
        rk182x = sum(j["device"] == "rk182x" for j in jobs)
        moved = sum(j["transfer_bytes"] for j in jobs)
        print(f"{policy:<16}{result['makespan_ns']:>13}{rk3588:>13}{rk182x:>13}{moved:>16}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
