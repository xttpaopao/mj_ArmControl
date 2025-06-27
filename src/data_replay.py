import h5py
import numpy as np

def load_mujoco_data(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        # 加载数据
        images = np.array(f["images"])
        joint_positions = np.array(f["joint_positions"])
        timestamps = np.array(f["timestamps"])
        
        # 加载元数据
        joint_names = [name.decode('utf-8') for name in f.attrs["joint_names"]]
        camera_name = f.attrs["camera_name"]
        image_size = f.attrs["image_size"]
        
        return {
            "images": images,
            "joint_positions": joint_positions,
            "timestamps": timestamps,
            "joint_names": joint_names,
            "camera_name": camera_name,
            "image_size": image_size
        }

# 使用示例
data = load_mujoco_data("mujoco_data.hdf5")
print(f"Loaded {len(data['images'])} frames")
print(f"Joint names: {data['joint_names']}")