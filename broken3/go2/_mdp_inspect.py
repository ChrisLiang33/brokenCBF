"""Throwaway inspector -- verifies what cbf_task.mdp actually contains
after the (Isaac Sim-loaded) interpreter has imported it.
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import re

import cbf_task.mdp as mdp

print()
print("=" * 70)
print("MDP MODULE INSPECTOR")
print("=" * 70)
print(f"module file: {mdp.__file__}")
print(f"file size:   {os.path.getsize(mdp.__file__)} bytes")
print()

src = open(mdp.__file__).read()
defs_in_file = re.findall(r"^def (\w+)", src, re.MULTILINE)
defs_on_module = [n for n in dir(mdp) if callable(getattr(mdp, n, None))
                  and not n.startswith("__")]

print(f"defs found by regex in file ({len(defs_in_file)}):")
for d in defs_in_file:
    print(f"  - {d}")
print()
print(f"callable attrs on loaded module ({len(defs_on_module)}):")
for d in defs_on_module:
    print(f"  - {d}")
print()

missing = [d for d in defs_in_file if not hasattr(mdp, d)]
print(f"defs in file but NOT on loaded module: {missing}")
print()

# specific check for the broken function
print(f"has _compute_lidar: {hasattr(mdp, '_compute_lidar')}")
print(f"has _ensure_post_physics: {hasattr(mdp, '_ensure_post_physics')}")
print(f"has lidar_obs: {hasattr(mdp, 'lidar_obs')}")

simulation_app.close()
