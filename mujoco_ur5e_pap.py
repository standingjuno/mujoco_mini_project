import time
import mujoco
import numpy as np
import mink

arm = mujoco.MjSpec.from_file(
    "/Users/standingjuno/MuJoCo/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
)
gripper = mujoco.MjSpec.from_file(
    "/Users/standingjuno/MuJoCo/mujoco_menagerie/robotiq_2f85/2f85.xml"
)
env = mujoco.MjSpec.from_file("pickandplace.xml")

arm.attach(gripper, prefix="gripper_", site="attachment_site")
env.attach(arm, prefix="robot_", site="robot_base_site")

model = env.compile()

# print("=== joint list check ===")
# for i in range(model.njnt):
#     name = model.joint(i).name
#     idx = model.jnt_qposadr[i]
#     print(f"  [{i}] {name}  (qpos_idx={idx})")

# print("=== actuator list check ===")
# for i in range(model.nu):
#     print(f"  [{i}] {model.actuator(i).name}")

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# ── init pose setting 
current_phase = "init"

init_qpos = np.deg2rad([0, -165, 115, -90, -90, 0])

joint_names = ["robot_shoulder_pan_joint", "robot_shoulder_lift_joint", "robot_elbow_joint",
               "robot_wrist_1_joint", "robot_wrist_2_joint", "robot_wrist_3_joint"]

actuator_names = ["robot_shoulder_pan", "robot_shoulder_lift", "robot_elbow",
                  "robot_wrist_1", "robot_wrist_2", "robot_wrist_3"]

for i, name in enumerate(joint_names):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[joint_id]] = init_qpos[i]

mujoco.mj_forward(model, data)

for i, name in enumerate(actuator_names):
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    data.ctrl[act_id] = init_qpos[i]

# ── IK
configuration = mink.Configuration(model)
configuration.update(data.qpos)

print(f"IK Starting joint angle (deg): {np.rad2deg(configuration.q[:6])}")

task = mink.FrameTask(
    frame_name="robot_gripper_pinch",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=1.0,
    gain=0.1, # IK가 목표를 향해 얼마나 빠르게 수렴할지 결정하는 값
)

box_pose = configuration.get_transform_frame_to_world("red_box_center", "site")
downward_rotation = mink.SO3.from_y_radians(np.pi)
pre_grasp = mink.SE3.from_rotation_and_translation(
    rotation=downward_rotation,
    translation=box_pose.translation() + np.array([0.0, 0.0, 0.2])
)

# ✅ IK를 반복 실행해서 목표 joint angle 계산 (시뮬 시작 전에 한 번만)
task.set_target(pre_grasp)
for _ in range(1000):  # 충분히 반복해서 수렴
    vel = mink.solve_ik(configuration, [task], dt=0.01, solver="quadprog")
    configuration.integrate_inplace(vel, 0.01)

target_qpos = np.array([
    configuration.q[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]]
    for name in joint_names
])

print(f"Targeting joint angle (deg): {np.rad2deg(target_qpos)}")

print(f"red_box 위치: {box_pose.translation()}")
print(f"로봇 베이스 위치: {data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'robot_base_site')]}")


# ── interpolation setting
MOVE_DURATION = 6.0   # 몇 초에 걸쳐 이동할지
interp_t = 0.0        # interpolation 진행도 (0.0 ~ 1.0)

def on_key(key):
    global current_phase, interp_t
    print(f"key pressed: {key}")
    if key == ord(" ") and current_phase == "init":
        current_phase = "pre_grasp"
        interp_t = 0.0
        print("trigger fire!")

with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    while viewer.is_running():
        step_start = time.time()

        if current_phase == "pre_grasp" and interp_t < 1.0:
            interp_t = min(interp_t + model.opt.timestep / MOVE_DURATION, 1.0)
            t = interp_t * interp_t * (3 - 2 * interp_t)
            current_ctrl = (1 - t) * init_qpos + t * target_qpos

            for i, name in enumerate(actuator_names):
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                data.ctrl[act_id] = current_ctrl[i]

        mujoco.mj_step(model, data)
        viewer.sync()
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)