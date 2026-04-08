import mujoco
import mujoco.viewer
import numpy as np
import sys
import termios
import tty

# ===== 키 입력 함수 =====
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ===== 모델 로드 =====
model = mujoco.MjModel.from_xml_path("/Users/standingjuno/MuJoCo/mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
data = mujoco.MjData(model)

print("=== Actuators ===")
for i in range(model.nu):
    print(i, model.actuator(i).name)

# ===== 설정 =====
step = 0.8
direction = np.ones(model.nu)  # 필요하면 방향 보정

print("\nControls:")
print("q/a: joint0")
print("w/s: joint1")
print("e/d: joint2")
print("r/f: joint3")
print("t/g: joint4")
print("y/h: joint5")
print("x: exit\n")

# ===== 시뮬레이션 =====
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        key = get_key()

        target = data.qpos.copy()

        if key == 'q':
            target[0] += step * direction[0]
        elif key == 'a':
            target[0] -= step * direction[0]

        elif key == 'w':
            target[1] += step * direction[1]
        elif key == 's':
            target[1] -= step * direction[1]

        elif key == 'e':
            target[2] += step * direction[2]
        elif key == 'd':
            target[2] -= step * direction[2]

        elif key == 'r':
            target[3] += step * direction[3]
        elif key == 'f':
            target[3] -= step * direction[3]

        elif key == 't':
            target[4] += step * direction[4]
        elif key == 'g':
            target[4] -= step * direction[4]

        elif key == 'y':
            target[5] += step * direction[5]
        elif key == 'h':
            target[5] -= step * direction[5]

        elif key == 'x':
            print("Exiting...")
            break

        # ctrl 적용
        data.ctrl[:] = target[:model.nu]

        mujoco.mj_step(model, data)
        viewer.sync()