import mujoco
from mujoco import viewer

model = mujoco.MjModel.from_xml_path("./urdf/Diana_7/urdf/mjmodel.xml")

data = mujoco.MjData(model)

duration = 10.0  # seconds
framerate = 60  # Hz
mujoco.mj_resetData(model, data)
data.ctrl = (0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0)
with viewer.launch_passive(model, data) as viewer:
    while viewer.is_running:
        # while data.time < duration:
        mujoco.mj_step(model, data)
        # print(data.qpos)
        viewer.sync()
