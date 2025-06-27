#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pink
import qpsolvers
import pinocchio as pin
import mujoco
import mujoco.viewer

from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector
from pink import solve_ik
from joyconrobotics import JoyconRobotics


init_gpos_left = [0.33, 0.21, 0.95,-1.5708,0,-1.5708]
init_gpos_right = [0.33, -0.21, 0.95,-1.5708,0,-1.5708]
joycon_left = JoyconRobotics("left",offset_position_m=init_gpos_left[:3],offset_euler_rad=init_gpos_left[3:])
joycon_right = JoyconRobotics("right",offset_position_m=init_gpos_right[:3],offset_euler_rad=init_gpos_right[3:])

class IKSolver:
    def __init__(self, mjcf_path: str):
        # 加载模型
        self.mjcf_path = mjcf_path
        self.robot = pin.RobotWrapper.BuildFromMJCF(self.mjcf_path)
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
            self.model.getJointId("L_WRIST_R"),
        ]
        self.right_arm_indices = [
            self.model.getJointId("R_SHOULDER_P"),
            self.model.getJointId("R_SHOULDER_R"),
            self.model.getJointId("R_SHOULDER_Y"),
            self.model.getJointId("R_ELBOW_Y"),
            self.model.getJointId("R_WRIST_P"),
            self.model.getJointId("R_WRIST_Y"),
            self.model.getJointId("R_WRIST_R"),
        ]

        # 初始化配置
        self.q_init = pin.neutral(self.model)

        # 创建任务
        self.left_task = FrameTask(
            self.left_ee_frame,
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.right_task = FrameTask(
            self.right_ee_frame,
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.posture_task = PostureTask(cost=1e-3)

    def dual_arm_IK(self, left_target_pose: np.ndarray, right_target_pose: np.ndarray):
        """求解双臂逆运动学"""
        # 处理输入，输入是三维坐标加上rpy角度
        left_target_trans = left_target_pose[:3]
        left_target_rpy = left_target_pose[3:]
        left_target_rot = pin.rpy.rpyToMatrix(left_target_rpy)

        right_target_trans = right_target_pose[:3]
        right_target_rpy = right_target_pose[3:]
        right_target_rot = pin.rpy.rpyToMatrix(right_target_rpy)

        tasks = [self.left_task, self.right_task, self.posture_task]

        
        configuration = pink.Configuration(self.model, self.data, self.q_init)
        for task in tasks:
            task.set_target_from_configuration(configuration)
        solver = qpsolvers.available_solvers[0]

        if "quadprog" in qpsolvers.available_solvers:
            solver = "quadprog"

        tolerance_position = 1e-4
        tolerance_orientation = 1e-3
        max_iterations = 500
        iteration = 0
        dt = 0.01
        while True:
            iteration += 1

            # left_side
            left_target = self.left_task.transform_target_to_world
            left_target.translation = left_target_trans
            left_target.rotation = left_target_rot

            # right_side
            right_target = self.right_task.transform_target_to_world
            right_target.translation = right_target_trans
            right_target.rotation = right_target_rot

            # compute errors
            # left_side
            left_current_pose = configuration.get_transform_frame_to_world(
                self.left_task.frame
            )
            left_target_pose = self.left_task.transform_target_to_world
            left_diff = left_target_pose.actInv(left_current_pose)

            left_position_error = left_diff.translation
            left_orientation_error = pin.log3(left_diff.rotation)
            left_position_error_norm = np.linalg.norm(left_position_error)
            left_orientation_error_norm = np.linalg.norm(left_orientation_error)

            #right_side
            right_current_pose = configuration.get_transform_frame_to_world(
                self.right_task.frame
            )
            right_target_pose = self.right_task.transform_target_to_world
            right_diff = right_target_pose.actInv(right_current_pose)

            right_position_error = right_diff.translation
            right_orientation_error = pin.log3(right_diff.rotation)
            right_position_error_norm = np.linalg.norm(right_position_error)
            right_orientation_error_norm = np.linalg.norm(right_orientation_error)
            

            if (
                left_position_error_norm < tolerance_position
                and left_orientation_error_norm < tolerance_orientation
                and right_position_error_norm < tolerance_position
                and right_orientation_error_norm < tolerance_orientation
            ):
                print("error within orientation")
                break
            if iteration >= max_iterations:
                print("\nReached maximum iterations")
                break
            velocity = solve_ik(configuration, tasks, dt, solver=solver)
            configuration.integrate_inplace(velocity, dt) 
            optimal_q = configuration.q.copy()

            return optimal_q


if __name__ == "__main__":
    ik = IKSolver("./urdf/T170D_with_hands/urdf/T170D_locked.xml")


    # 初始化模型
    model = mujoco.MjModel.from_xml_path("./urdf/T170D_with_hands/urdf/scene.xml")
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            left_pose, left_gripper, left_control_button = joycon_left.get_control()
            right_pose,right_gripper,right_control_button = joycon_right.get_control()
            q = ik.dual_arm_IK(np.array(left_pose),np.array(right_pose))
            
            
            viewer.sync()
