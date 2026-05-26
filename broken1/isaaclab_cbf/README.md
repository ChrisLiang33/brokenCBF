# CBF + Adaptive Params on Go2 (Isaac Lab port of lastChance/index.py MVP)

## Goal
Port the 2D MVP to a 3D Isaac Lab simulation of a Unitree Go2 quadruped.
The RL policy outputs the 5 CBF parameters (α, φ, a, b, c); the CBF filter
takes a nominal velocity from a planner and produces a safe velocity, which
is sent to a frozen pre-trained Go2 locomotion policy.

## Architectures to compare
**Arch 1 (`fullsetup`, RMA-style):**
- `priv_obs` (true obstacle states, true uncertainty levels) → encoder → `z`
- proprioception (joint pos/vel, IMU, body vel)
- past actions (last K CBF param outputs)
- LiDAR occupancy grid → CNN
- All concatenated → MLP head → 5 CBF params

**Arch 2 (`nopriv`):**
- Identical except `priv_obs` is zeroed before the encoder (and the encoder
  is dropped).
- Ablation to measure how much the privileged signal helps.

## Pipeline (one env step)
```
planner.step()           → u_nom (vx, vy)  in body frame
policy.forward(obs)      → (α, φ, a, b, c)
safety_filter(...)       → u_safe
locomotion.step(u_safe)  → joint targets
sim.step()               → next obs
```

## Files
- `core/safety_filter.py` — batched smooth-SDF CBF, Picard for b·||u||
- `core/perception.py`    — batched lidar→occupancy + circle fitting
- `core/scenes.py`        — scene definitions (open, spath, corridor, slalom, narrow, gauntlet)
- `policy/networks.py`    — TeacherPolicy and StudentPolicy
- `env/go2_cbf_env.py`    — Isaac Lab env (skeleton — fill in USD references)
- `train.py`              — PPO entry point (rsl_rl)
- `eval.py`               — evaluation matching the 2D MVP (collision/unsafe/reach/time/detour/min_h)

## "World isn't perfect" signals (sim version)
| MVP signal | Isaac Lab equivalent |
|---|---|
| `noise_std` on velocity | `events.randomize_actuator_gains` + observation noise |
| `obs_drift_std` | scripted obstacle motion via `RigidObjectCfg` velocity setters |
| `g_eps` | `events.randomize_rigid_body_mass` + actuator randomization |
| `adv_prob` | Wrap planner with probabilistic "aim at nearest obstacle" |

## Running
On the labbox (assumes Isaac Lab installed):
```
cd isaaclab_cbf
# Train fullsetup (with priv encoder):
python train.py --arch fullsetup --task Isaac-Go2-CBF-v0 --num_envs 4096
# Train nopriv (ablation):
python train.py --arch nopriv --task Isaac-Go2-CBF-v0 --num_envs 4096
# Evaluate both vs baselines:
python eval.py --fullsetup_ckpt fullsetup.pt --nopriv_ckpt nopriv.pt
```

## Status
This is a starting skeleton. Critical next steps:
1. Wire `env/go2_cbf_env.py` to a real Isaac Lab Go2 USD scene.
2. Plug in a pre-trained locomotion policy (Isaac Lab provides Go2 Velocity policy).
3. Define obstacles as RigidObjects in the scene config.
4. Connect RayCaster sensor → `core/perception.py`.
5. Run `train.py` and iterate.
