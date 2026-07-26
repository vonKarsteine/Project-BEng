# Building & Running the ROS Half (Ubuntu 20.04 + ROS Noetic)

The `catkin_ws/src` tree in this folder is complete and self-contained.
It cannot be compiled on this Windows machine; copy it to an Ubuntu 20.04
box (the thesis drone's Intel NUC, or a VM for simulation).

## 1. System dependencies

```bash
# ROS Noetic desktop-full (see wiki.ros.org/noetic/Installation/Ubuntu), then:
sudo apt install -y libarmadillo-dev libpcl-dev ros-noetic-pcl-ros \
    ros-noetic-cv-bridge ros-noetic-octomap ros-noetic-octomap-server octovis \
    ros-noetic-mavros ros-noetic-mavros-extras python3-catkin-tools
sudo bash /opt/ros/noetic/lib/mavros/install_geographiclib_datasets.sh
```

Livox SDK (needed by `livox_ros_driver`):

```bash
git clone https://github.com/Livox-SDK/Livox-SDK.git
cd Livox-SDK/build && cmake .. && make -j && sudo make install
```

## 2. Build

```bash
mkdir -p ~/catkin_ws
cp -r <this folder>/catkin_ws/src ~/catkin_ws/
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
# message packages first avoids generation races:
catkin_make --pkg quadrotor_msgs livox_ros_driver traj_utils
catkin_make
source devel/setup.bash
```

Notes:
- `uav_simulator/local_sensing`: CPU rendering is fine — keep
  `ENABLE_CUDA false` in its CMakeLists.txt (default).
- `odom_visualization` needs `libarmadillo-dev` (installed above).
- FAST-LIO's CMake defaults to `Debug` build type but forces `-O3`; for
  onboard use you may set `CMAKE_BUILD_TYPE=Release`.

## 3. Smoke tests (in order)

### 3a. Pure planner simulation (no hardware)

```bash
roslaunch ego_planner single_run_in_sim.launch
# rviz opens; use "2D Nav Goal" to command flights through the random forest
```

### 3b. FAST-LIO on a recorded Mid-360 bag

```bash
roslaunch fast_lio mapping_mid360.launch
rosbag play your_mid360.bag
# check: /Odometry (nav_msgs/Odometry), /cloud_registered (PointCloud2)
```

### 3c. Integrated FAST-LIO + EGO-Planner (thesis Ch. 4/5)

```bash
roslaunch ego_planner run_with_fastlio.launch
rosbag play your_mid360.bag       # or live LiDAR
# grid_map occupancy builds from /cloud_registered in rviz;
# set goals with "2D Nav Goal"
```

Defaults: 12x12x2 m map, 0.1 m resolution, 0.3 m inflation, max_vel 2 m/s
(thesis values for the ~5x5 m room). Tune via launch args.

## 4. Real flight (QAV250 + Pixhawk4-mini + NUC + Mid-360)

1. **Mid-360 driver**: the real Mid-360 needs
   [Livox-SDK2](https://github.com/Livox-SDK/Livox-SDK2) +
   [livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2)
   (driver1 does not talk to Mid-360 hardware; it is only needed to *build*
   FAST-LIO's message types / replay bags). To run this FAST-LIO snapshot on
   driver2, rename in a copy of the package:
   - `src/preprocess.h:4,90,106`, `src/preprocess.cpp:44,88`,
     `src/laserMapping.cpp:59,301`: `livox_ros_driver` → `livox_ros_driver2`
     (include + namespace)
   - `CMakeLists.txt:54`, `package.xml`: dependency → `livox_ros_driver2`
   (Alternative: upgrade `fast_lio` to current upstream master, which
   supports driver2 out of the box with the same config layout.)

2. **MAVROS + vision pose**:
   ```bash
   roslaunch mavros px4.launch fcu_url:=/dev/ttyACM0:921600
   roslaunch ego_planner run_with_fastlio.launch use_mavros:=true realworld_experiment:=true
   ```
   `lidar_to_mavros` republishes `/Odometry` → `mavros/vision_pose/pose`
   at 30 Hz and prints estimated vs PX4 pose side by side — **verify they
   agree before arming**.

3. **PX4 EKF2 parameters** (external vision):
   - PX4 ≥ 1.14: `EKF2_EV_CTRL = 11` (pos+vel+yaw), `EKF2_HGT_REF = 3` (vision)
   - PX4 1.13: `EKF2_AID_MASK = 24`, `EKF2_HGT_MODE = 3`
   - Start the drone **level** (FAST-LIO's `camera_init` frame = initial IMU
     pose; the bridge forwards it unchanged as ENU). If the LiDAR is mounted
     off-center, set `EKF2_EV_POS_X/Y/Z`.

4. **Offboard control**: `traj_server` outputs
   `quadrotor_msgs/PositionCommand` on `drone_0_planning/pos_cmd`. A
   PositionCommand → PX4 offboard-setpoint bridge is required; the canonical
   reference for exactly this stack is
   [ZJU-FAST-Lab/Fast-Drone-250](https://github.com/ZJU-FAST-Lab/Fast-Drone-250)
   (`px4ctrl`), or the XTDrone Python bridge used in the thesis simulation.

## 5. Map extraction for RL (thesis Ch. 5)

```bash
# FAST-LIO saves PCD/scans.pcd when pcd_save_en: true (set in config/mid360.yaml)
rosrun pcd2octomap pcd2octomap scans.pcd map.bt 0.05
octovis map.bt                                  # inspect

# 2D occupancy raster for the LwH training loop (works on Windows too):
python3 <this folder>/scripts/pcd_to_raster.py scans.pcd --out map_raster \
        --resolution 0.1 --zmin 0.3 --zmax 1.8
```

## Known gaps / notes

- The thesis's modified `random_forest_sensing.cpp` (publishing a map from a
  saved pcd) was never on disk; `scripts/pcd_to_raster.py` covers that use.
- `px4_gazebo_fuel-master` in the original folder is an empty husk (failed
  extraction). For the Gazebo/XTDrone simulation of Ch. 4, use
  [XTDrone](https://github.com/robin-shaun/XTDrone) directly.
- `grid_map.cpp` still subscribes `/vins_fusion/extrinsic` (stock EGO-Swarm);
  harmless without a publisher — the LiDAR cloud path does not use it.
