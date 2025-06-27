import numpy as np
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
import mujoco
import mujoco.viewer

class T170DIKSolver:
    def __init__(self, xml_path):
        """初始化双手机器人IK求解器"""
        # 加载模型
        self.robot = pin.RobotWrapper.BuildFromMJCF("./urdf/T170D_with_hands/urdf/T170D_with_hands.xml")
        self.model = self.robot.model
        self.data = self.robot.data
        
        # 定义末端执行器
        self.left_ee_frame = "L_gripper_link"
        self.right_ee_frame = "R_gripper_link"
        self.left_ee_id = self.model.getFrameId(self.left_ee_frame)
        self.right_ee_id = self.model.getFrameId(self.right_ee_frame)
        
        # 关节分组
        self.left_arm_indices = [
            self.model.getJointId("L_SHOULDER_P"),
            self.model.getJointId("L_SHOULDER_R"),
            self.model.getJointId("L_SHOULDER_Y"),
            self.model.getJointId("L_ELBOW_Y"),
            self.model.getJointId("L_WRIST_P"),
            self.model.getJointId("L_WRIST_Y"),
            self.model.getJointId("L_WRIST_R")
        ]
        self.right_arm_indices = [
            self.model.getJointId("R_SHOULDER_P"),
            self.model.getJointId("R_SHOULDER_R"),
            self.model.getJointId("R_SHOULDER_Y"),
            self.model.getJointId("R_ELBOW_Y"),
            self.model.getJointId("R_WRIST_P"),
            self.model.getJointId("R_WRIST_Y"),
            self.model.getJointId("R_WRIST_R")
        ]
        
        # 初始化配置
        self.q_init = pin.neutral(self.model)
        
    def solve_dual_arm_ik(self, left_target_pose, right_target_pose):
        """求解双臂逆运动学"""
        # 创建任务
        left_task = FrameTask(
            self.left_ee_frame, 
            position_cost=1.0, orientation_cost=1.0
        )
        right_task = FrameTask(
            self.right_ee_frame,
            position_cost=1.0, orientation_cost=1.0
        )
        
        # 设置目标
        left_task.set_target(left_target_pose)
        right_task.set_target(right_target_pose)
        
        # 可选：添加姿态任务防止奇异构型
        posture_task = PostureTask(
            cost=0.1  # 较低权重
        )
        
        # 构建问题
        problem = pink.IKProblem(
            model=self.model,
            tasks=[left_task, right_task, posture_task],
            q_init=self.q_init,
            active_joints=self.left_arm_indices + self.right_arm_indices
        )
        
        # 求解IK
        result = solve_ik(problem, max_iter=100, tol=1e-4)
        
        return result.q, result.success

    def render_in_mujoco(self, q, duration=5.0):
        """在MuJoCo中渲染结果"""
        model = mujoco.MjModel.from_xml_path("./urdf/T170D_with_hands/urdf/scene.xml")
        data = mujoco.MjData(model)
        
        # 设置关节状态
        data.qpos[:len(q)] = q
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            while viewer.is_running() and (time.time() - start_time) < duration:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.001)

if __name__ == "__main__":
    import time
    
    # 初始化IK求解器
    ik_solver = T170DIKSolver("./urdf/T170D_with_hands/urdf/T170D_with_hands.xml")
    
    # 定义目标位姿 (SE3)
    left_target_pose = pin.SE3(
        pin.Quaternion(0.707, 0.707, 0, 0),  # 绕X轴旋转90度
        np.array([0.5, 0.2, 0.8])            # 位置
    )
    right_target_pose = pin.SE3(
        pin.Quaternion(0.707, -0.707, 0, 0), # 绕X轴旋转-90度
        np.array([0.5, -0.2, 0.8])
    )
    
    # 求解IK
    q_sol, success = ik_solver.solve_dual_arm_ik(left_target_pose, right_target_pose)
    
    if success:
        print("IK求解成功！关节角度：")
        print(q_sol)
        
        # 提取双臂关节角度
        left_angles = q_sol[ik_solver.left_arm_indices]
        right_angles = q_sol[ik_solver.right_arm_indices]
        print(f"左臂关节: {left_angles}")
        print(f"右臂关节: {right_angles}")
        
        # 可视化
        ik_solver.render_in_mujoco(q_sol, duration=10.0)
    else:
        print("IK求解失败！")