# Revision Log

Every change relative to the original sources, with the reason. "Original"
means the partial code that shipped with the project (`LwH-master`, `planner`,
`FAST_LIO-main` — local reference copies, not part of this repository) and the
upstream repos listed under Provenance.

Publication notes: the pristine upstream clones (`vendor/`) and FAST-LIO's
upstream demo media (`catkin_ws/src/fast_lio/doc/`, 98 MB of GIFs) are not
tracked in this repository — provenance URLs below are authoritative. License
files travel with each vendored subtree (`fast_lio/LICENSE` GPL-2.0,
`ego_planner/LICENSE` + `uav_simulator/LICENSE` GPL-3.0,
`livox_ros_driver/LICENSE.txt` MIT).

## Provenance / vendored sources

| What | Source | Why |
|---|---|---|
| `gym-uav/` | https://github.com/DennisWangCW/gym-uav (master, single commit 2024-02-21) | The `gym_uav` package (`uav-v0`) was never shipped with LwH; this is the original author's own release. The student's 2024 `__pycache__` files prove they used exactly this code. |
| LwH upstream reference | https://github.com/DennisWangCW/LwH | The local `Learn-with-Helper` is byte-identical to this repo (which was published broken — see its issues #1-#4). |
| `catkin_ws/src/fast_lio/include/ikd-Tree/` | https://github.com/hku-mars/ikd-Tree branch `fast_lio` | The GitHub ZIP of FAST-LIO ships an **empty** submodule dir → build blocker. |
| `catkin_ws/src/uav_simulator/` | https://github.com/ZJU-FAST-Lab/ego-planner-swarm (master) | The on-disk `planner/` was only the `src/planner` half of EGO-Swarm; the simulator half (incl. the hard compile-dependency `quadrotor_msgs`) was missing. Upstream master is code-identical to the on-disk 2022-11 vintage (last code commit 2021-11). |
| `catkin_ws/src/livox_ros_driver/` | https://github.com/Livox-SDK/livox_ros_driver | FAST-LIO compile dependency (`livox_ros_driver/CustomMsg`). |

## A. gym-uav patches

| File | Change | Why |
|---|---|---|
| `setup.py` | `install_requires` `gym` → `gymnasium`; added `packages=` | The env is written against gymnasium; declaring `gym` would pull the wrong package. |
| `gym_uav/__init__.py` | added registration `uav-v0` → `UavDenseEnv`, `max_episode_steps=2000` | LwH trains on the id `uav-v0`; sparse reward is the `Config` default. 2000 matches LwH `--max-episode-length`. |
| `gym_uav/envs/uav_dense_env.py` `step` | truncation `== 100` → `>= Config.max_episode_steps` (2000) | Internal cap was hardcoded to 100 steps; thesis episodes are 2000 steps. |
| `gym_uav/envs/uav_dense_env.py` `reset` | `np.random.seed = seed` → `np.random.seed(seed)` (guarded `seed >= 0`) | Original **assigned over** the seed function — a bug that also silently killed all later seeding. |
| `gym_uav/envs/utils.py` `Config` | added `max_episode_steps = 2000` | Backs the fix above. |

## B. Learn-with-Helper fixes

Blockers (the original could not run at all — `../LwH-master/logsuav-v0_log.txt`
shows the student's 2024-03 run died before the first episode):

| File | Change | Why |
|---|---|---|
| `policy_domain.py` | removed `from ddpg import DDPG`; `agent_ddpg = None`; final `else` branch raises `NotImplementedError` | The `DDPG` class was never published (LwH issue #1). It was constructed unconditionally, breaking *every* run even with `--use-prior` off. The thesis only uses `demo_type` `uav`/`uav_wrong`. |
| `policy_domain.py` | Naive-Pri distance decode constant `3000` → `1440` | `state[9]` is normalized by `sqrt(2)*period*num_circle` = `sqrt(2)*1440` in the vendored env. |
| `ddpg.py` | 500-episode Pendulum training + matplotlib at module import quarantined under `if __name__ == '__main__'` | Importing it (as `policy_domain` used to) started a full training run in all 17 processes. |
| `main.py` | added `--demo-type` (default `uav`) | README documents the flag; `policy_domain.py` reads `args.demo_type`; it never existed in `main.py`. |
| `main.py` | `--variance` default `[0.1225]` | Had no default → `TypeError` crash. The arg is a **variance**; thesis σ_h = 0.35 ⇒ 0.1225. Measured helper success ≈ 50% at 0.1225 vs ≈ 10% at 0.35. |
| `main.py` | optimizer `if/if` → `if/elif/else raise`; `makedirs(save_model_dir)`; removed dead `import gym` | `NameError` on unknown optimizer; save dir never created. |
| `environment.py` | port to gymnasium: `gym.make(..., disable_env_checker=True)`, seed via `env.unwrapped.seed()`, `reset` tuple unpack, `step` 5-tuple → old 4-tuple (`done or truncated`, `bool()` around the env's int `done`) | gymnasium ≥ 0.26 API. |

Version rot (numpy ≥ 1.24 / torch ≥ 2.0):

| File | Change |
|---|---|
| `shared_optim.py` | all scalar-first `add_/addcmul_/addcdiv_` overloads → keyword `alpha=`/`value=` forms (removed in torch 1.5+/2.x) |
| `train.py` | `np.float` → `float` |
| `test.py` | `np.int`/`.astype(np.int)` → `int` |
| `gym_eval.py` | removed `torch.set_default_tensor_type` and `gym.wrappers.Monitor` (both gone) |

Correctness bugs (would train silently wrong or mislead the evaluation):

| File | Change | Why |
|---|---|---|
| `utils.py` `ensure_shared_grads` | assign `shared_param._grad = param.grad` **every** call (original early-returned once set) | Under torch ≥ 2.0 `zero_grad()` frees grad tensors by default, so the original one-time aliasing left the shared model training on the first iteration's gradients forever. Paired with `zero_grad(set_to_none=False)` in `train.py`. |
| `train.py` | removed per-worker zeroing of the shared step/episode counters | Every worker reset the *global* counters at startup — racy with 16 workers; the parent already initializes them. |
| `test.py` `Model_Buffer.put` | `clone()` each state_dict tensor | The state_dict references the live shared-memory tensors; every queued "snapshot" aliased the newest weights, so evaluations were mislabeled. |
| `test.py` | `'is_success' in info` → `info.get('is_success', False)` | The env sets the key with value `False` on crashes/truncations too; the original counted every crash as a success. |
| `test.py` | log path and checkpoint path via `os.path.join`; checkpoints now go to `--save-model-dir` | Original wrote `logsuav-v0_log.txt` into the CWD (missing separator) and saved checkpoints where `gym_eval.py` could not find them. |
| `test.py` | termination: break when the last producible snapshot label (`training_steps // cache_interval * cache_interval`) has been evaluated | Original condition `training_steps > args.training_steps` could never fire (labels are floored to cache-interval multiples) → the tester hung forever after training finished. |
| `gym_eval.py` | `range(1)` → `range(--num-episodes)`; `--render` is now opt-in; prints per-episode result and final success rate | Original evaluated exactly 1 episode regardless of the flag and always tried to open the vtk window. |
| `train.py`/`test.py` | `setproctitle` optional; `torch.set_num_threads(1)` | Windows friendliness; avoids thread oversubscription across 8-16 worker processes. |

New files: `ddpg_baseline.py` (classical DDPG on `uav-v0`, 400/300 nets per
`policy_domain.Config`, logs in the same format — the thesis Fig 10 comparison
curve), `../scripts/plot_results.py`, `../scripts/run_*.ps1`.

Known leftovers (intentional): `--l2-regular` and `--stack-frames` are parsed
but unused (vestiges of a pre-release supervised phase visible in the 2019
`.pyc` files); the shared step counter is a frozen `nn.Linear(1,1)` abused as a
cross-process scalar — kept, it is load-bearing.

## C. ROS half

| File | Change | Why |
|---|---|---|
| `ego_planner/plan_manage/launch/simulator.xml` | `c_num/p_num/min_dist` args given defaults | Declared required but no caller passes them; only the CDATA-disabled `random_forest` block consumes them. Defaults remove the abort-on-re-enable trap. |
| `ego_planner/plan_manage/launch/advanced_param.xml` | `virtual_ceil_height` → `virtual_ceil_yp`/`virtual_ceil_yn`; added `odom_depth_timeout` | This grid_map vintage reads `virtual_ceil_yp/yn` (`grid_map.cpp:43-44`), never `virtual_ceil_height`; `odom_depth_timeout` was read but never set. (Note: both ceil params are parsed but not enforced in this vintage.) |
| `ego_planner/plan_manage/launch/advanced_param_exp.xml` | added args `pose_type` (default 2) and `realworld_experiment` (default true); added `odom_depth_timeout` | Makes the exp param file reusable by the new FAST-LIO launch without changing historical behavior. |
| `ego_planner/plan_manage/launch/run_with_fastlio.launch` | **new** | The FAST-LIO → EGO-Planner integration the thesis describes (odom `/Odometry`, cloud `/cloud_registered`, static TF `world→camera_init`, indoor params, max_vel 2 m/s, rviz 2D Nav Goal, optional mavros bridge). No such wiring existed anywhere in the original code. |
| `lidar_to_mavros/` | **new package** wrapping the fixed `../planner/lidar_to_mavros.cpp` | Original was a loose, uncompilable file: undeclared `pi` (→ `M_PI`), missing `<iomanip>`, unused `tf` include, duplicate `firstPoseReceived` branches collapsed, `frame_id` set. |
| `pcd2octomap/` | **new package** | The thesis names the `pcd2octomap` executable (Ch. 5) but no implementation was on disk. |
| `fast_lio/` | unchanged except ikd-Tree fill-in | Upstream FAST-LIO is correct as-is. |

Not changed on purpose: all EGO-Swarm C++ (byte-stock upstream, including the
`/vins_fusion/extrinsic` subscription — harmless without a publisher, and the
LiDAR path does not use the extrinsic), `single_run_in_exp.launch`
(historical VINS+RealSense reference).
