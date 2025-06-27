import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation

def get_ee_pose(model, data, body_name):
    """获取指定身体的位姿（位置 + 四元数）"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos = data.body(body_id).xpos  # 位置 [x,y,z]
    quat = data.body(body_id).xquat  # 四元数 [w,x,y,z]
    return pos, quat

def quat_to_rpy(quat):
    """四元数转RPY欧拉角（弧度）"""
    r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # xyzw -> wxyz
    return r.as_euler('xyz')

# 初始化模型
model = mujoco.MjModel.from_xml_path("./urdf/T170D_with_hands/urdf/T170D_locked.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        
        # 获取左右手腕位姿
        left_pos, left_quat = get_ee_pose(model, data, "L_gripper_link")
        right_pos, right_quat = get_ee_pose(model, data, "R_gripper_link")
        
        # 转换为RPY
        left_rpy = quat_to_rpy(left_quat)
        right_rpy = quat_to_rpy(right_quat)
        
        # 实时打印（或发送到其他程序）
        print(f"Left EE - Pos: {left_pos} | RPY: {left_rpy}")
        print(f"Right EE - Pos: {right_pos} | RPY: {right_rpy}")
        
        viewer.sync()
