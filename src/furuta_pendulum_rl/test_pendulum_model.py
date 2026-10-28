import mujoco
from mujoco import viewer
from furuta_pendulum_rl import constants

model = mujoco.MjModel.from_xml_path(constants.PENDULUM_MODEL_PATH)
data = mujoco.MjData(model)

viewer.launch(model, data)