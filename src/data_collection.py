import mujoco
import mujoco.viewer
import numpy as np
import cv2
import h5py
import time
from collections import deque

class MujocoDataRecorder:
    def __init__(self, model_path, hdf5_path, max_steps=10000, image_size=(640, 480)):
        """
        初始化数据记录器
        :param image_size: (width, height) 格式的图像尺寸
        """
        # 初始化模型和数据
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 确保渲染器使用正确的尺寸 (width, height)
        self.renderer = mujoco.Renderer(self.model, width=image_size[0], height=image_size[1])
        
        # 存储设置
        self.hdf5_path = hdf5_path
        self.max_steps = max_steps
        self.current_step = 0
        self.image_size = image_size  # 保持 (width, height) 格式
        
        # 数据缓冲区
        self.image_buffer = deque(maxlen=max_steps)
        self.joint_buffer = deque(maxlen=max_steps)
        self.time_buffer = deque(maxlen=max_steps)
        
        # 查找所有关节
        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.joint_indices = [self.model.joint(name).id for name in self.joint_names]
        
        # 查找相机
        self.camera_name = "rgb_camera"  # 修改为你的相机名称
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        
        # 创建HDF5文件结构
        with h5py.File(hdf5_path, 'w') as f:
            # 图像数据集 shape: (frames, height, width, channels)
            # 注意：虽然我们使用(width, height)初始化，但MuJoCo返回的是(height, width)
            f.create_dataset("images", 
                           shape=(0, image_size[1], image_size[0], 3),  # (height, width)
                           maxshape=(None, image_size[1], image_size[0], 3),
                           dtype=np.uint8,
                           compression="gzip")
            
            f.create_dataset("joint_positions",
                            shape=(0, len(self.joint_indices)),
                            maxshape=(None, len(self.joint_indices)),
                            dtype=np.float32,
                            compression="gzip")
            
            f.create_dataset("timestamps",
                           shape=(0,),
                           maxshape=(None,),
                           dtype=np.float64,
                           compression="gzip")
            
            # 保存元数据
            f.attrs["joint_names"] = np.array(self.joint_names, dtype='S')
            f.attrs["camera_name"] = self.camera_name
            f.attrs["model_path"] = model_path
            f.attrs["image_size"] = image_size  # 存储为 (width, height)
    
    def record_step(self):
        """记录一步数据"""
        # 模拟一步
        mujoco.mj_step(self.model, self.data)
        
        # 渲染图像 (返回的是height, width, channels)
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb_image = self.renderer.render()
        
        # 获取关节位置
        joint_positions = self.data.qpos[self.joint_indices].copy()
        
        # 记录时间戳
        timestamp = time.time()
        
        # 存入缓冲区
        self.image_buffer.append(rgb_image)
        self.joint_buffer.append(joint_positions)
        self.time_buffer.append(timestamp)
        
        self.current_step += 1
        
        return rgb_image
    
    def save_to_hdf5(self):
        """将缓冲区数据保存到HDF5文件"""
        with h5py.File(self.hdf5_path, 'a') as f:
            # 获取当前数据集大小
            current_size = f["images"].shape[0]
            new_size = current_size + len(self.image_buffer)
            
            # 调整数据集大小
            f["images"].resize(new_size, axis=0)
            f["joint_positions"].resize(new_size, axis=0)
            f["timestamps"].resize(new_size, axis=0)
            
            # 直接保存图像数据 (已经是height, width顺序)
            f["images"][current_size:new_size] = np.array(self.image_buffer)
            f["joint_positions"][current_size:new_size] = np.array(self.joint_buffer)
            f["timestamps"][current_size:new_size] = np.array(self.time_buffer)
            
        # 清空缓冲区
        self.image_buffer.clear()
        self.joint_buffer.clear()
        self.time_buffer.clear()
        
        print(f"Saved {new_size - current_size} frames to HDF5")
    
    def run_simulation(self, duration=10.0, save_interval=100):
        """运行模拟并记录数据"""
        start_time = time.time()
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                while viewer.is_running() and (time.time() - start_time) < duration:
                    # 记录一步数据
                    rgb_image = self.record_step()
                    
                    # 显示图像 (OpenCV使用height, width顺序)
                    cv2.imshow("Camera View", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    # 定期保存数据
                    if self.current_step % save_interval == 0:
                        self.save_to_hdf5()
                    
                    # 同步查看器
                    viewer.sync()
            
            # 保存剩余数据
            if len(self.image_buffer) > 0:
                self.save_to_hdf5()
        
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        recorder = MujocoDataRecorder(
            model_path="./urdf/T170D_with_hands/urdf/T170D_with_hands.xml",
            hdf5_path="mujoco_data.hdf5",
            max_steps=10000,
            image_size=(640, 480)  # (width, height)
        )
        recorder.run_simulation(duration=30.0)
    except Exception as e:
        print(f"Error occurred: {str(e)}")