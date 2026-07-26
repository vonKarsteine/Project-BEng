# UAV Indoor Exploration & Path Planning — BEng Project (Revised, Working Edition)

Complete, runnable revision of the BEng project *"UAV-based Exploration and
Path Planning for Indoor Environments with Artificial Intelligence"*
(Harbin Institute of Technology, Shenzhen, 2024): a deep-reinforcement-learning
trajectory planner — **LwH** (A3C with a decaying non-expert *helper* prior,
fused as a precision-weighted product of Gaussians) — trained in a
sparse-reward UAV navigation environment, plus the **FAST-LIO 2** (Livox
Mid-360) + **EGO-Planner** ROS stack and the PX4/MAVROS glue used for real
indoor flight.

The public sources the project builds on were partial or broken (the
simulation environment was never released with LwH, the MAVROS glue node did
not compile, the simulator half of the planner workspace was missing, plus
assorted version rot and latent bugs). This repository is the completed
revision: every change and every piece of provenance is recorded in
[REVISIONS.md](REVISIONS.md).

| Thesis chapter | Component | Where | Status |
|---|---|---|---|
| Ch. 3 — LwH DRL algorithm | A3C + helper prior (`Learn with Helper`) | `Learn-with-Helper/` | **runs & trains** (Windows or Linux, CPU) |
| Ch. 3 — simulation env | `gym_uav` (`uav-v0`, sparse reward) | `gym-uav/` | vendored from its original author + patched |
| Ch. 4 — SLAM | FAST-LIO 2 (Livox Mid-360) | `catkin_ws/src/fast_lio/` | complete (ikd-Tree filled in); build on Ubuntu |
| Ch. 4 — planning | EGO-Planner (EGO-Swarm) | `catkin_ws/src/ego_planner/` + `uav_simulator/` | complete; build on Ubuntu |
| Ch. 4/5 — integration | FAST-LIO → EGO-Planner wiring | `.../plan_manage/launch/run_with_fastlio.launch` | **new** (described in the report; code never existed) |
| Ch. 5 — PX4 glue | `/Odometry` → `mavros/vision_pose/pose` | `catkin_ws/src/lidar_to_mavros/` | fixed + packaged (original didn't compile) |
| Ch. 5 — map for RL | pcd → octomap / 2D raster | `catkin_ws/src/pcd2octomap/`, `scripts/pcd_to_raster.py` | new tools |

## Setup (DRL part)

```bash
conda create -n lwh python=3.11 -y
conda activate lwh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "gymnasium==0.29.1" "numpy<2" vtk matplotlib pandas setproctitle
pip install -e ./gym-uav
```

## Quick start (DRL part)

```bash
# smoke: short training run (~1 min)
cd Learn-with-Helper
python main.py --workers 2 --training-steps 20000 --cache-interval 2000 --test-episodes 5 --use-prior --log-dir logs/smoke
```

Reproduction of the report's Fig. 10 comparison (each run ~10–20 min on 8
cores; PowerShell launchers — set `$python` at the top to your env's
interpreter):

```powershell
scripts\run_lwh_prior.ps1        # LwH (A3C + SenAvo-Pri prior)
scripts\run_a3c_noprior.ps1      # baseline without prior
scripts\run_ddpg_baseline.ps1    # classical DDPG baseline
scripts\plot_fig10.ps1           # overlay curves -> results\fig10_reproduction.png
```

Evaluate a trained policy (add `--render` for the vtk visualization):

```bash
python gym_eval.py --env uav-v0 --num-episodes 50
```

Key facts recovered during the revision:

- The missing simulation environment is the LwH author's own
  [gym-uav](https://github.com/DennisWangCW/gym-uav) (vendored in `gym-uav/`,
  registered here as `uav-v0`, sparse reward by default, episode cap raised
  100 → 2000 to match the training setup).
- `--variance` is a **variance**: the report's σ_h = 0.35 corresponds to
  `--variance 0.1225`. Measured helper success rate at that value ≈ 50%
  (a working low-performance prior); at 0.35 it degrades to ≈ 10%.
- The full change log against the original sources is in
  [REVISIONS.md](REVISIONS.md).

## ROS part (target: Ubuntu 20.04 + ROS Noetic)

`catkin_ws/src` is a complete, self-contained workspace source tree.
Follow [BUILD.md](BUILD.md) for the build sequence, bag-replay smoke tests,
the integrated FAST-LIO + EGO-Planner launch, and the real-flight checklist
(PX4 EKF2 parameters included).

## Folder map

```
.
├── README.md            <- this file
├── REVISIONS.md         <- every change vs the original sources, with reasons
├── BUILD.md             <- Ubuntu build & run guide for the ROS half
├── Learn-with-Helper/   <- fixed LwH code (Ch. 3)
├── gym-uav/             <- vendored + patched uav-v0 environment
├── scripts/             <- training launchers, Fig-10 plotter, pcd->raster
├── results/             <- plots land here
└── catkin_ws/src/       <- complete ROS workspace (Ch. 4/5)
```

## References & third-party components

Each vendored component keeps its own license file in its folder; the
detailed provenance (repos, commits, what was changed and why) is in
[REVISIONS.md](REVISIONS.md).

- Project report: *UAV-based Exploration and Path Planning for Indoor
  Environments with Artificial Intelligence*, B.Eng. thesis, Harbin Institute
  of Technology (Shenzhen), 2024.
- **LwH / gym-uav** — C. Wang, J. Wang, J. Wang, X. Zhang,
  "Deep-Reinforcement-Learning-Based Autonomous UAV Navigation With Sparse
  Rewards," *IEEE Internet of Things Journal* 7(7), 2020.
  Code: [DennisWangCW/LwH](https://github.com/DennisWangCW/LwH),
  [DennisWangCW/gym-uav](https://github.com/DennisWangCW/gym-uav).
- **FAST-LIO 2** — W. Xu, Y. Cai, D. He, J. Lin, F. Zhang, "FAST-LIO2: Fast
  Direct LiDAR-Inertial Odometry," *IEEE Transactions on Robotics*, 2022.
  Code: [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO) (GPL-2.0),
  with [hku-mars/ikd-Tree](https://github.com/hku-mars/ikd-Tree).
- **EGO-Planner / EGO-Swarm** — X. Zhou et al., "EGO-Planner: An ESDF-Free
  Gradient-Based Local Planner for Quadrotors," *IEEE RA-L*, 2021, and
  "EGO-Swarm: A Fully Autonomous and Decentralized Quadrotor Swarm System in
  Cluttered Environments," *ICRA*, 2021.
  Code: [ZJU-FAST-Lab/ego-planner-swarm](https://github.com/ZJU-FAST-Lab/ego-planner-swarm) (GPL-3.0).
- **livox_ros_driver** —
  [Livox-SDK/livox_ros_driver](https://github.com/Livox-SDK/livox_ros_driver) (MIT).
