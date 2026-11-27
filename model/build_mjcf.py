"""Generate the MJCF model from phase0_params.yaml. Never hand edit the XML.

Conventions (docs/conventions.md): x forward, y left, z up. Every joint axis is +y.
Positive hip and knee swing the foot backward. Positive wheel rotation rolls forward.
Left and right legs use identical signs.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import math 

from tools.params import load_params, leg_ik

OUT_PATH = Path(__file__).resolve().parent / "generated" / "biped.xml"

BODY_PART = "2"
WHEEL = "4"
FLOOR_AFFINITY = "6"
CTRL_HEADROOM = 1.15   # torque mode clamp above peak, so motor constant randomization above 1.0 has effect


def fmt(*values):
    return " ".join(f"{v:.6g}" for v in values)


def check_inertia(name, mass, diag):
    a, b, c = diag
    if mass <= 0 or min(diag) <= 0:
        raise ValueError(f"{name}: mass and inertia must be positive")
    if a + b < c or a + c < b or b + c < a:
        raise ValueError(f"{name}: inertia violates the triangle inequality")


def validate(p):
    for key in ("body", "thigh", "shank", "wheel"):
        check_inertia(key, p[key]["mass_kg"], p[key]["inertia_kgm2"])
    for joint in ("hip", "knee"):
        lo, hi = p["joints"][joint]["range_rad"]
        if lo >= hi:
            raise ValueError(f"{joint}: range must be increasing")
    if p["wheel"]["geom_type"] not in ("cylinder", "sphere"):
        raise ValueError("wheel.geom_type must be 'cylinder' or 'sphere'")


def build_xml(p, mode = "position"):
    if mode not in ("position", "torque"):
        raise ValueError(f"unknown actuator mode {mode}")
    validate(p)
    sim, b, th, sh, wh, sl, ar = p["sim"], p["body"], p["thigh"], p["shank"], p["wheel"], p["shell"], p["arm"]
    jt, lm, wm, ct = p["joints"], p["leg_motor"], p["wheel_motor"], p["contact"]
    hip_y = b["hip_separation_m"] / 2
    com_h = b["com_height_above_hips_m"]
    z0 = wh["radius_m"] + th["length_m"] + sh["length_m"] + sim["spawn_clearance_m"]
    leg_armature = lm["rotor_inertia_kgm2"] * lm["gear_ratio"] ** 2
    wheel_armature = wm["rotor_inertia_kgm2"] * wm["gear_ratio"] ** 2

    root = ET.Element("mujoco", model=p["meta"]["name"])
    root.append(ET.Comment(" GENERATED FROM phase0_params.yaml. DO NOT EDIT. "))
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root, "option",
        timestep=fmt(sim["physics_dt_s"]),
        gravity=fmt(0, 0, -sim["gravity_mps2"]),
        integrator="implicitfast",
    )

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world, "geom", name="floor", type="plane", size="5 5 0.1",
        friction=fmt(*ct["floor_friction"]), contype="0", conaffinity=FLOOR_AFFINITY,
        rgba="0.85 0.85 0.85 1",
    )

    torso = ET.SubElement(world, "body", name="torso", pos=fmt(0, 0, z0))
    ET.SubElement(torso, "freejoint", name="root")
    ET.SubElement(
        torso, "inertial", pos=fmt(0, 0, com_h),
        mass=fmt(b["mass_kg"]), diaginertia=fmt(*b["inertia_kgm2"]),
    )
    ET.SubElement(
        torso, "geom", name="torso_geom", type="box",
        size=fmt(*[s / 2 for s in b["size_m"]]), pos=fmt(0, 0, com_h),
        contype=BODY_PART, conaffinity="0", rgba="0.55 0.6 0.75 1",
    )
    ET.SubElement(torso, "site", name="imu", pos=fmt(0, 0, com_h), size="0.005")
    # D08 shell rails and the locked arm placeholder. Mass only for now; the
    # inertia is inferred from the geometry, so no explicit inertial element.
    for side, ysign in (("l", 1), ("r", -1)):
        rail = ET.SubElement(
            torso, "body", name=f"shell_{side}",
            pos=fmt(0, ysign * sl["y_offset_m"], sl["z_offset_m"]),
        )
        ET.SubElement(
            rail, "geom", name=f"shell_{side}_geom", type="capsule",
            fromto=fmt(-sl["length_m"] / 2, 0, 0, sl["length_m"] / 2, 0, 0),
            size=fmt(sl["side_radius_m"]), mass=fmt(sl["mass_kg"] / 2),
            contype=BODY_PART, conaffinity="0", rgba="0.3 0.3 0.6 1",
        )

    # Arm is pure mass with no contact. The real arm collision geometry arrives
    # with the Phase 3 CAD.
    arm_body = ET.SubElement(torso, "body", name="arm_mass", pos=fmt(0, 0, ar["com_above_hips_m"]))
    ET.SubElement(
        arm_body, "geom", name="arm_mass_geom", type="sphere", size=fmt(0.02),
        mass=fmt(ar["mass_kg"]), contype="0", conaffinity="0",
        rgba="0.6 0.3 0.3 1",
    )
    if sl.get("top_radius_m"):
        half_w = b["size_m"][1] / 2
        ET.SubElement(
            torso, "geom", name="shell_top_geom", type="cylinder",
            fromto=fmt(0, -half_w, sl["top_z_m"], 0, half_w, sl["top_z_m"]),
            size=fmt(sl["top_radius_m"]), mass="0",
            contype=BODY_PART, conaffinity="0", rgba="0.3 0.3 0.6 0.4",
        )
    for side, sy in (("l", hip_y), ("r", -hip_y)):
        thigh = ET.SubElement(torso, "body", name=f"thigh_{side}", pos=fmt(0, sy, 0))
        ET.SubElement(
            thigh, "joint", name=f"hip_{side}", type="hinge", axis="0 1 0",
            range=fmt(*jt["hip"]["range_rad"]), damping=fmt(jt["damping_Nms_per_rad"]),
            frictionloss=fmt(jt["frictionloss_Nm"]), armature=fmt(leg_armature),
        )
        ET.SubElement(
            thigh, "inertial", pos=fmt(0, 0, -th["length_m"] / 2),
            mass=fmt(th["mass_kg"]), diaginertia=fmt(*th["inertia_kgm2"]),
        )
        ET.SubElement(
            thigh, "geom", name=f"thigh_{side}_geom", type="capsule",
            fromto=fmt(0, 0, 0, 0, 0, -th["length_m"]), size=fmt(th["radius_m"]),
            contype=BODY_PART, conaffinity="0", rgba="0.4 0.4 0.4 1",
        )

        shank = ET.SubElement(thigh, "body", name=f"shank_{side}", pos=fmt(0, 0, -th["length_m"]))
        ET.SubElement(
            shank, "joint", name=f"knee_{side}", type="hinge", axis="0 1 0",
            range=fmt(*jt["knee"]["range_rad"]), damping=fmt(jt["damping_Nms_per_rad"]),
            frictionloss=fmt(jt["frictionloss_Nm"]), armature=fmt(leg_armature),
        )
        ET.SubElement(
            shank, "inertial", pos=fmt(0, 0, -sh["length_m"] / 2),
            mass=fmt(sh["mass_kg"]), diaginertia=fmt(*sh["inertia_kgm2"]),
        )
        ET.SubElement(
            shank, "geom", name=f"shank_{side}_geom", type="capsule",
            fromto=fmt(0, 0, 0, 0, 0, -sh["length_m"]), size=fmt(sh["radius_m"]),
            contype=BODY_PART, conaffinity="0", rgba="0.5 0.5 0.5 1",
        )

        wheel = ET.SubElement(shank, "body", name=f"wheel_{side}", pos=fmt(0, 0, -sh["length_m"]))
        ET.SubElement(
            wheel, "joint", name=f"wheel_{side}", type="hinge", axis="0 1 0",
            damping=fmt(jt["wheel_damping_Nms_per_rad"]), armature=fmt(wheel_armature),
        )
        
        ET.SubElement(
            wheel, "inertial", pos="0 0 0",
            mass=fmt(wh["mass_kg"]), diaginertia=fmt(*wh["inertia_kgm2"]),
        )
        if wh["geom_type"] == "cylinder":
            ET.SubElement(
                wheel, "geom", name=f"wheel_{side}_geom", type="cylinder",
                fromto=fmt(0, -wh["width_m"] / 2, 0, 0, wh["width_m"] / 2, 0),
                size=fmt(wh["radius_m"]), friction=fmt(*ct["wheel_friction"]),
                contype=WHEEL, conaffinity="0", rgba="0.1 0.1 0.1 1",
            )
        else:
            ET.SubElement(
                wheel, "geom", name=f"wheel_{side}_geom", type="sphere",
                size=fmt(wh["radius_m"]), friction=fmt(*ct["wheel_friction"]),
                contype=WHEEL, conaffinity="0", rgba="0.1 0.1 0.1 1",
            )

    topology = p.get("legs", {}).get("topology", "serial")
    belt = mode == "torque" and topology == "coaxial_belt"
    if belt:
        tendons = ET.SubElement(root, "tendon")
        for side in ("l", "r"):
            fx = ET.SubElement(tendons, "fixed", name=f"belt_{side}")
            ET.SubElement(fx, "joint", joint=f"hip_{side}", coef="1")
            ET.SubElement(fx, "joint", joint=f"knee_{side}", coef="1")

    # Self collision between the legs and the body (D27). Body parts otherwise collide only with
    # the floor. MJX cannot collide a cylinder with a box, so the wheels are checked against the
    # shell geoms, which cover the torso box from the sides and above.
    contact = ET.SubElement(root, "contact")
    body_geoms = ["torso_geom"] + [g.get("name") for g in root.iter("geom")
                                   if g.get("name", "").startswith("shell_")]
    for side in ("l", "r"):
        for g in body_geoms:
            ET.SubElement(contact, "pair", geom1=f"shank_{side}_geom", geom2=g, condim="1")
            if g != "torso_geom":
                ET.SubElement(contact, "pair", geom1=f"wheel_{side}_geom", geom2=g, condim="1")

    # Position mode is a kinematic debug model: joint space PD regardless of topology.
    act = ET.SubElement(root, "actuator")
    for side in ("l", "r"):
        for jn in ("hip", "knee"):
            lo, hi = jt[jn]["range_rad"]
            if mode == "position":
                ET.SubElement(
                    act, "position", name=f"{jn}_{side}_pos", joint=f"{jn}_{side}",
                    kp=fmt(lm["kp_Nm_per_rad"]), kv=fmt(lm["kv_Nms_per_rad"]),
                    ctrlrange=fmt(lo, hi),
                    forcerange=fmt(-lm["peak_torque_Nm"], lm["peak_torque_Nm"]),
                )
            else:
                lim = lm["peak_torque_Nm"] * CTRL_HEADROOM
                target = ({"tendon": f"belt_{side}"} if (belt and jn == "knee")
                          else {"joint": f"{jn}_{side}"})
                ET.SubElement(act, "motor", name=f"{jn}_{side}_torque", gear="1",
                              ctrlrange=fmt(-lim, lim), **target)
        wlim = wm["peak_torque_Nm"] * CTRL_HEADROOM
        ET.SubElement(
            act, "motor", name=f"wheel_{side}_torque", joint=f"wheel_{side}", gear="1",
            ctrlrange=fmt(-wlim, wlim),
        )

    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "framequat", name="orientation", objtype="site", objname="imu")
    ET.SubElement(sensors, "gyro", name="gyro", site="imu")
    ET.SubElement(sensors, "accelerometer", name="accel", site="imu")
    # Phase 2 keyframes. The lying poses are approximate spawn states; the check
    # scripts settle them under gravity before measuring anything.
    H_nom = p["pose"]["nominal_hip_height_m"]
    q1, q2 = leg_ik(p, H_nom)
    H_lo = p["pose"]["height_cmd_range_m"][0]
    c1, c2 = leg_ik(p, H_lo)
    clr = sim["spawn_clearance_m"]
    s2 = math.sqrt(0.5)

    def qp(z, quat, qa, qb):
        return fmt(0, 0, z, *quat, qa, qb, 0, qa, qb, 0)

    def ct(qa, qb):
        return fmt(0, 0, 0, 0, 0, 0) if mode == "torque" else fmt(qa, qb, 0, qa, qb, 0)

    kf = ET.SubElement(root, "keyframe")
    for name, z, quat, qa, qb in (
        ("stand",       H_nom + clr, (1, 0, 0, 0),    q1, q2),
        ("crouch",      H_lo + clr,  (1, 0, 0, 0),    c1, c2),
        ("lying_front", 0.15,        (s2, 0, s2, 0),  c1, c2),
        ("lying_back",  0.15,        (s2, 0, -s2, 0), c1, c2),
        ("lying_side",  0.15,        (s2, s2, 0, 0),  c1, c2),
    ):
        ET.SubElement(kf, "key", name=name, qpos=qp(z, quat, qa, qb), ctrl=ct(qa, qb))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["position", "torque"], default="position")
    args = ap.parse_args()
    xml = build_xml(load_params(), mode=args.mode)
    out = OUT_PATH if args.mode == "position" else OUT_PATH.with_name("biped_torque.xml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    print("wrote", out)


if __name__ == "__main__":
    main()
