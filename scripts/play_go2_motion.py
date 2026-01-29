import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

MENAGERIE_DIR = Path("mujoco_menagerie")
GO2_XML = MENAGERIE_DIR / "unitree_go2" / "scene.xml"

QPOS_PATH = Path("retarget_out/go2_qpos.npy")
DT = 0.02  # playback dt (use motion FrameDuration if you want exact)

def main():
    qpos_traj = np.load(QPOS_PATH)
    model = mujoco.MjModel.from_xml_path(str(GO2_XML))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for t in range(qpos_traj.shape[0]):
                data.qpos[:] = qpos_traj[t]
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(DT)

if __name__ == "__main__":
    main()
