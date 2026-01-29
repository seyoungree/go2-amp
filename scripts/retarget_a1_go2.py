import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mujoco


MENAGERIE_DIR = Path("mujoco_menagerie")  # cloned repo folder
A1_XML = MENAGERIE_DIR / "unitree_a1" / "scene.xml"
GO2_XML = MENAGERIE_DIR / "unitree_go2" / "scene.xml"

AMP_HW_DIR = Path("AMP_for_hardware")  # cloned repo folder
MOTION_FILE = AMP_HW_DIR / "datasets" / "mocap_motions" / "walk.txt"  # change to trot.txt etc if present

OUT_DIR = Path("retarget_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# IK tuning
IK_ITERS = 25
IK_DAMPING = 1e-2
IK_STEP_SCALE = 0.7

# Quaternion format in motion file:
#   - "wxyz" is common in AMP pipelines
#   - if your results look wrong, set this to "xyzw"
ROOT_QUAT_FORMAT = "wxyz"  # or "xyzw"


# =========================
# Helpers
# =========================
def _name_to_id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    _id = mujoco.mj_name2id(model, objtype, name)
    if _id < 0:
        raise ValueError(f"Name not found: {name}")
    return _id


def get_actuated_joint_names(model: mujoco.MjModel) -> List[str]:
    """Actuator -> joint mapping; returns names in actuator order."""
    out = []
    for a in range(model.nu):
        jid = int(model.actuator_trnid[a][0])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname:
            out.append(jname)
    return out


def joint_qpos_adr(model: mujoco.MjModel, joint_name: str) -> int:
    jid = _name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[jid])


def set_hinge_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float):
    adr = joint_qpos_adr(model, joint_name)
    data.qpos[adr] = value


def quat_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    if q.shape != (4,):
        raise ValueError("quat must be shape (4,)")
    if ROOT_QUAT_FORMAT == "wxyz":
        return q
    if ROOT_QUAT_FORMAT == "xyzw":
        # xyzw -> wxyz
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    raise ValueError("ROOT_QUAT_FORMAT must be 'wxyz' or 'xyzw'")


def quat_wxyz_to_mat(qwxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = map(float, qwxyz)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def load_amp_motion_file(path: str) -> Tuple[float, np.ndarray]:
    with open(path, "r") as f:
        d = json.load(f)
    dt = float(d.get("FrameDuration", 0.02))
    frames = np.asarray(d["Frames"], dtype=np.float32)
    return dt, frames


def parse_frame(frame: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Typical AMP frame layout:
      root_pos (3)
      root_rot (4)
      dof_pos (12)
      root_lin_vel (3)
      root_ang_vel (3)
      dof_vel (12)
      ... optional extras
    """
    if frame.shape[0] < 37:
        raise ValueError(f"Frame too short: {frame.shape[0]}")
    return {
        "root_pos": frame[0:3],
        "root_rot": frame[3:7],
        "dof_pos": frame[7:19],
        "root_lin_vel": frame[19:22],
        "root_ang_vel": frame[22:25],
        "dof_vel": frame[25:37],
    }


def find_foot_sites(model: mujoco.MjModel) -> List[int]:
    """Heuristic: pick site names containing 'foot' or 'toe'."""
    cands = []
    for sid in range(model.nsite):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if not nm:
            continue
        low = nm.lower()
        if ("foot" in low) or ("toe" in low):
            cands.append(sid)
    if len(cands) < 4:
        raise RuntimeError(f"Could not find 4 foot/toe sites; found {len(cands)}. Set explicit site names.")
    # stable subset
    cands = sorted(cands, key=lambda sid: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid))[:4]
    return cands


def label_feet_by_pose(model: mujoco.MjModel, data: mujoco.MjData, site_ids: List[int]) -> Dict[str, int]:
    """
    Assign FR/FL/RR/RL by world x(front/back) and y(left/right) at current pose.
    Convention: +y is left in MuJoCo.
    """
    pts = []
    for sid in site_ids:
        pos = data.site_xpos[sid].copy()
        pts.append((sid, pos[0], pos[1]))

    pts_sorted_x = sorted(pts, key=lambda t: t[1], reverse=True)
    front = pts_sorted_x[:2]
    rear = pts_sorted_x[2:]

    fr = min(front, key=lambda t: t[2])[0]
    fl = max(front, key=lambda t: t[2])[0]
    rr = min(rear, key=lambda t: t[2])[0]
    rl = max(rear, key=lambda t: t[2])[0]
    return {"FR": fr, "FL": fl, "RR": rr, "RL": rl}


def feet_local_positions(model: mujoco.MjModel, data: mujoco.MjData, feet: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Foot positions in base frame (local to root)."""
    base_pos = data.qpos[0:3].copy()
    base_quat = data.qpos[3:7].copy()
    R = quat_wxyz_to_mat(base_quat)
    Rt = R.T
    out = {}
    for leg, sid in feet.items():
        p_w = data.site_xpos[sid].copy()
        out[leg] = Rt @ (p_w - base_pos)
    return out


def build_joint_index_map(model: mujoco.MjModel, joint_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    qpos_adrs, dof_adrs = [], []
    for jn in joint_names:
        jid = _name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qpos_adrs.append(int(model.jnt_qposadr[jid]))
        dof_adrs.append(int(model.jnt_dofadr[jid]))
    return np.asarray(qpos_adrs, dtype=int), np.asarray(dof_adrs, dtype=int)


def solve_ik_go2(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: List[str],
    feet: Dict[str, int],
    target_local: Dict[str, np.ndarray],
):
    """
    Damped least squares IK on 12 hinge joints.
    Root pose stays at whatever is currently in data.qpos[0:7].
    """
    qpos_adrs, dof_adrs = build_joint_index_map(model, joint_names)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    base_pos = data.qpos[0:3].copy()
    base_quat = data.qpos[3:7].copy()
    R = quat_wxyz_to_mat(base_quat)
    target_world = {k: base_pos + R @ target_local[k] for k in target_local.keys()}

    for _ in range(IK_ITERS):
        mujoco.mj_forward(model, data)

        r_list, J_list = [], []
        for leg in ["FR", "FL", "RR", "RL"]:
            sid = feet[leg]
            p = data.site_xpos[sid].copy()
            r = (target_world[leg] - p)
            r_list.append(r)

            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
            J_list.append(jacp[:, dof_adrs])

        r_stack = np.concatenate(r_list, axis=0)     # (12,)
        J_stack = np.concatenate(J_list, axis=0)     # (12,12)

        if np.linalg.norm(r_stack) < 1e-4:
            break

        A = J_stack.T @ J_stack + IK_DAMPING * np.eye(J_stack.shape[1])
        b = J_stack.T @ r_stack
        dq = np.linalg.solve(A, b)

        data.qpos[qpos_adrs] += IK_STEP_SCALE * dq

    mujoco.mj_forward(model, data)


def main():
    # Load models
    a1_model = mujoco.MjModel.from_xml_path(str(A1_XML))
    a1_data = mujoco.MjData(a1_model)
    mujoco.mj_resetData(a1_model, a1_data)
    mujoco.mj_forward(a1_model, a1_data)

    go2_model = mujoco.MjModel.from_xml_path(str(GO2_XML))
    go2_data = mujoco.MjData(go2_model)
    mujoco.mj_resetData(go2_model, go2_data)
    mujoco.mj_forward(go2_model, go2_data)

    # Foot sites + labeling
    a1_feet = label_feet_by_pose(a1_model, a1_data, find_foot_sites(a1_model))
    go2_feet = label_feet_by_pose(go2_model, go2_data, find_foot_sites(go2_model))
    print("A1 feet:", {k: mujoco.mj_id2name(a1_model, mujoco.mjtObj.mjOBJ_SITE, v) for k, v in a1_feet.items()})
    print("Go2 feet:", {k: mujoco.mj_id2name(go2_model, mujoco.mjtObj.mjOBJ_SITE, v) for k, v in go2_feet.items()})

    # Actuated joints (use first 12)
    a1_joints = get_actuated_joint_names(a1_model)[:12]
    go2_joints = get_actuated_joint_names(go2_model)[:12]
    if len(a1_joints) < 12 or len(go2_joints) < 12:
        raise RuntimeError(f"Expected 12 actuated joints. Got A1={len(a1_joints)} Go2={len(go2_joints)}")
    print("A1 joints:", a1_joints)
    print("Go2 joints:", go2_joints)

    # Load motion frames
    dt, frames = load_amp_motion_file(str(MOTION_FILE))
    T = frames.shape[0]
    print(f"Loaded motion {MOTION_FILE}  frames={frames.shape}  dt={dt}")

    go2_qpos = np.zeros((T, go2_model.nq), dtype=np.float32)
    go2_qvel = np.zeros((T, go2_model.nv), dtype=np.float32)

    prev_qpos = None

    # Retarget per frame
    for t in range(T):
        f = parse_frame(frames[t])

        # Apply A1 pose for FK
        a1_data.qpos[0:3] = f["root_pos"]
        a1_data.qpos[3:7] = quat_to_wxyz(f["root_rot"])

        for i, jn in enumerate(a1_joints):
            set_hinge_qpos(a1_model, a1_data, jn, float(f["dof_pos"][i]))

        mujoco.mj_forward(a1_model, a1_data)

        # Feet in base frame
        a1_local = feet_local_positions(a1_model, a1_data, a1_feet)

        # Solve IK on Go2 (root stays at default pose)
        solve_ik_go2(go2_model, go2_data, go2_joints, go2_feet, a1_local)

        go2_qpos[t] = go2_data.qpos.copy().astype(np.float32)

        # Approx qvel: hinge finite-diff only; base vel zeros (starter)
        if prev_qpos is not None:
            dq = (go2_qpos[t] - prev_qpos) / dt
            qv = np.zeros(go2_model.nv, dtype=np.float32)
            for jn in go2_joints:
                jid = _name_to_id(go2_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                qpos_adr = int(go2_model.jnt_qposadr[jid])
                dof_adr = int(go2_model.jnt_dofadr[jid])
                qv[dof_adr] = dq[qpos_adr]
            go2_qvel[t] = qv

        prev_qpos = go2_qpos[t].copy()

        if (t + 1) % 200 == 0:
            print(f"retargeted {t+1}/{T}")

    np.save(OUT_DIR / "go2_qpos.npy", go2_qpos)
    np.save(OUT_DIR / "go2_qvel.npy", go2_qvel)
    np.save(OUT_DIR / "go2_ref_obs.npy", np.concatenate([go2_qpos, go2_qvel], axis=1))
    print("Saved outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
