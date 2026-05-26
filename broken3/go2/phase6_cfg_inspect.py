"""Tiny inspect tool -- prints what our runner cfg actually looks like
at runtime so we can see whether `policy` is MISSING or some default
instance, what's on `actor`/`critic`, etc.

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase6_cfg_inspect.py
"""
from __future__ import annotations
import os, sys

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
AppLauncher(args_cli)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.agents import rma_actor_critic  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

cfg = load_cfg_from_registry("Isaac-CBF-Adaptive-Go2-RandObs-v0",
                              "rsl_rl_cfg_entry_point")

print()
print("=" * 78)
print("  RUNNER CFG INSPECT")
print("=" * 78)

for attr in ["class_name", "policy", "actor", "critic",
             "obs_groups", "empirical_normalization", "algorithm"]:
    if hasattr(cfg, attr):
        v = getattr(cfg, attr)
        type_name = type(v).__name__
        is_missing = (v is None or repr(v) == "<MISSING_TYPE>"
                      or str(v) == "MISSING"
                      or (hasattr(v, "__name__") and v.__name__ == "MISSING"))
        print(f"  {attr:>26}:  type={type_name:>30}  missing? {is_missing}")
        if attr == "actor" and not is_missing:
            print(f"    actor.class_name = {getattr(v, 'class_name', 'NONE')}")
        if attr == "critic" and not is_missing:
            print(f"    critic.class_name = {getattr(v, 'class_name', 'NONE')}")
        if attr == "policy" and not is_missing:
            print(f"    policy.class_name = {getattr(v, 'class_name', 'NONE')}")
            print(f"    policy = {v}")
    else:
        print(f"  {attr:>26}:  ATTRIBUTE MISSING (cfg doesn't have it)")
print("=" * 78)

import sys; sys.exit(0)
