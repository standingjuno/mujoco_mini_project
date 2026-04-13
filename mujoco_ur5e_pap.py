from re import L
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

init_qpos = np.deg2rad([0, -90, 90, -90, -90, 0])

joint_names = ["robot_shoulder_pan_joint", "robot_shoulder_lift_joint", "robot_elbow_joint",
               "robot_wrist_1_joint", "robot_wrist_2_joint", "robot_wrist_3_joint"]

actuator_names = ["robot_shoulder_pan", "robot_shoulder_lift", "robot_elbow",
                  "robot_wrist_1", "robot_wrist_2", "robot_wrist_3"]

for i, name in enumerate(joint_names):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[joint_id]] = init_qpos[i]

for i, name in enumerate(actuator_names):
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    data.ctrl[act_id] = init_qpos[i]

mujoco.mj_forward(model, data)

# --
configuration = mink.Configuration(model) # mink가 IK를 풀기 위해 model의 Joint, Position 등을 관리하는 객체 생성
configuration.update(data.qpos) # 현재 joint 각도를 configuration에 반영해야 IK가 현재 로봇 자세를 기준으로 계산

# IK가 달성해야 할 목표를 정의
task = mink.FrameTask(
    frame_name="robot_gripper_pinch",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=0.3,
    gain=0.1, # IK가 목표를 향해 얼마나 빠르게 수렴할지 결정하는 값
)

box_pose = configuration.get_transform_frame_to_world("red_box_center", "site")
downward_rotation = mink.SO3.from_x_radians(np.pi)
pre_grasp = mink.SE3.from_rotation_and_translation(
    rotation=downward_rotation,
    translation=box_pose.translation() + np.array([0.0, 0.0, 0.2])
)
task.set_target(pre_grasp)

for _ in range(1000):  
    vel = mink.solve_ik(configuration, [task], dt=0.01, solver="quadprog")
    configuration.integrate_inplace(vel, 0.01)

target_qpos = np.array([
    configuration.q[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]]
    for name in joint_names
])

# -- Gripper Control
gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot_gripper_fingers_actuator")
data.ctrl[gripper_act_id] = 0.0 # open == 0.0 & close == 255.0

# ── interpolation setting
MOVE_DURATION = 5.0   # 몇 초에 걸쳐 이동할지
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

        # red box 위로 이동
        if current_phase == "pre_grasp" and interp_t < 1.0:
            interp_t = min(interp_t + model.opt.timestep / MOVE_DURATION, 1.0)
            t = interp_t * interp_t * (3 - 2 * interp_t)
            current_ctrl = (1 - t) * init_qpos + t * target_qpos

            for i, name in enumerate(actuator_names):
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                data.ctrl[act_id] = current_ctrl[i]
        
        # red box 위로 이동 후 다음 target pose 계산
        elif current_phase == "pre_grasp" and interp_t >= 1.0:
            current_phase = "grasp"
            interp_t = 0.0
            print("pre_grasp 완료 → grasp 시작")

            grasp = mink.SE3.from_rotation_and_translation(
                rotation=downward_rotation,
                translation=box_pose.translation() + np.array([0.0, 0.0, 0.02])
            )
            task.set_target(grasp)

            grasp_configuration = mink.Configuration(model)
            grasp_configuration.update(data.qpos)
            for _ in range(1000):
                vel = mink.solve_ik(grasp_configuration, [task], dt=0.01, solver="quadprog")
                grasp_configuration.integrate_inplace(vel, 0.01)

            grasp_qpos = np.array([
                grasp_configuration.q[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]]
                for name in joint_names
            ])
        
        # red box를 잡기 위해 높이 조절
        elif current_phase == "grasp" and interp_t < 1.0:
            interp_t = min(interp_t + model.opt.timestep / MOVE_DURATION, 1.0)
            t = interp_t * interp_t * (3 - 2 * interp_t)
            current_ctrl = (1 - t) * target_qpos + t * grasp_qpos

            for i, name in enumerate(actuator_names):
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                data.ctrl[act_id] = current_ctrl[i]

        # red box를 잡기 위해 높이 조절 후 gripper 닫을 준비
        elif current_phase == "grasp" and interp_t >= 1.0:
            current_phase = "grip"
            interp_t = 0.0
            print("grasp 완료 → grip 시작 (그리퍼 천천히 닫기)")

        # 그리퍼 제어
        elif current_phase == "grip" and interp_t < 1.0:
            interp_t = min(interp_t + model.opt.timestep / MOVE_DURATION, 1.0)
            t = interp_t * interp_t * (3 - 2 * interp_t)
            data.ctrl[gripper_act_id] = t * 255.0  # 0 → 255 천천히

        # 그리퍼 제어 후 lift 준비
        elif current_phase == "grip" and interp_t >= 1.0:
            current_phase = "lift"
            interp_t = 0.0
            print("grip 완료 → lift 시작 (pre_grasp로 복귀)")

        # lift
        elif current_phase == "lift" and interp_t < 1.0:
            interp_t = min(interp_t + model.opt.timestep / MOVE_DURATION, 1.0)
            t = interp_t * interp_t * (3 - 2 * interp_t)
            current_ctrl = (1 - t) * grasp_qpos + t * target_qpos

            for i, name in enumerate(actuator_names):
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                data.ctrl[act_id] = current_ctrl[i]
        
        # lift 제어 후 180도 회전 준비
        elif current_phase == "lift" and interp_t >= 1.0:
            current_phase = "rotate"
            interp_t = 0.0
            print("lift 완료 → rotate 시작 (180도 회전)")

            rotate_target = mink.SE3.from_rotation_and_translation(
                rotation=downward_rotation,
                translation=box_pose.translation() + np.array([0.0, 0.0, 0.2])
            )

            rotate_configuration = mink.Configuration(model)
            rotate_configuration.update(data.qpos)

            # shoulder_pan만 180도 추가
            rotate_task = mink.FrameTask(
                frame_name="robot_gripper_pinch",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.3,
                gain=0.1,
            )
            rotate_task.set_target(rotate_target)

            rotate_init_qpos = target_qpos.copy()
            rotate_init_qpos[0] += np.pi  # shoulder_pan +180도

            rotate_qpos = rotate_init_qpos
            print(f"rotate joint angle (deg): {np.rad2deg(rotate_qpos)}")

        # 180도 회전
        elif current_phase == "rotate" and interp_t < 1.0:
            interp_t = min(interp_t + model.opt.timestep / MOVE_DURATION, 1.0)
            t = interp_t * interp_t * (3 - 2 * interp_t)
            current_ctrl = (1 - t) * target_qpos + t * rotate_qpos

            for i, name in enumerate(actuator_names):
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                data.ctrl[act_id] = current_ctrl[i]

        # 180도 회전 후 내릴 준비
        elif current_phase == "rotate" and interp_t >= 1.0:
            current_phase = "place"
            interp_t = 0.0
            print("rotate 완료 → place 시작 (박스 내려놓기)")

            configuration.update(data.qpos)
            current_ee = configuration.get_transform_frame_to_world("robot_gripper_pinch", "site")
            place_target = mink.SE3.from_rotation_and_translation(
                rotation=downward_rotation,
                translation=current_ee.translation() + np.array([0.0, 0.0, -0.15])  # 현재 위치에서 아래로
            )

            place_configuration = mink.Configuration(model)
            place_configuration.update(data.qpos)
            place_task = mink.FrameTask(
                frame_name="robot_gripper_pinch",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.3,
                gain=0.1,
            )
            place_task.set_target(place_target)
            for _ in range(1000):
                vel = mink.solve_ik(place_configuration, [place_task], dt=0.01, solver="quadprog")
                place_configuration.integrate_inplace(vel, 0.01)

            place_qpos = np.array([
                place_configuration.q[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]]
                for name in joint_names
            ])
            print(f"place joint angle (deg): {np.rad2deg(place_qpos)}")

        # 박스 내려놓기
        elif current_phase == "place" and interp_t < 1.0:
            interp_t = min(interp_t + model.opt.timestep / MOVE_DURATION, 1.0)
            t = interp_t * interp_t * (3 - 2 * interp_t)
            current_ctrl = (1 - t) * rotate_qpos + t * place_qpos

            for i, name in enumerate(actuator_names):
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                data.ctrl[act_id] = current_ctrl[i]

        # 박스 내려놓기 후 그리퍼 열 준비
        elif current_phase == "place" and interp_t >= 1.0:
            current_phase = "release"
            interp_t = 0.0
            print("place 완료 → release 시작 (그리퍼 열기)")

        # 그리퍼 열기
        elif current_phase == "release" and interp_t < 1.0:
            interp_t = min(interp_t + model.opt.timestep / MOVE_DURATION, 1.0)
            t = interp_t * interp_t * (3 - 2 * interp_t)
            data.ctrl[gripper_act_id] = (1 - t) * 255.0  # 255 → 0 천천히

        # 그리퍼 연 후 init pose로 갈 준비
        elif current_phase == "release" and interp_t >= 1.0:
            current_phase = "init_return"
            interp_t = 0.0
            print("release 완료 → init pose로 복귀")

        # init pose로 이동
        elif current_phase == "init_return" and interp_t < 1.0:
            interp_t = min(interp_t + model.opt.timestep / MOVE_DURATION, 1.0)
            t = interp_t * interp_t * (3 - 2 * interp_t)
            current_ctrl = (1 - t) * place_qpos + t * init_qpos

            for i, name in enumerate(actuator_names):
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                data.ctrl[act_id] = current_ctrl[i]

        elif current_phase == "init_return" and interp_t >= 1.0:
            current_phase = "done"
            print("✅ pick and place 완료!")

        mujoco.mj_step(model, data)
        viewer.sync()
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)