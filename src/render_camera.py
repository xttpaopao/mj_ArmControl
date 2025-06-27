import mujoco
import mujoco.viewer
import numpy as np
import cv2

model = mujoco.MjModel.from_xml_path("./urdf/T170D_with_hands/urdf/T170D_with_hands.xml")
data = mujoco.MjData(model)

# 启动被动查看器
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 创建另一个渲染器用于相机视图
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # 设置相机（可以使用名称或ID）
    camera_name = "rgb_camera"  # 你的相机名称
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # 更新并渲染相机视图
        renderer.update_scene(data, camera=camera_id)
        rgb_image = renderer.render()
        # 将RGB转换为BGR（OpenCV默认格式）
        camera_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # 显示相机图像
        cv2.imshow("TOP CAMERA", camera_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
