"""Generate the MJCF model from phase0_params.yaml. Never hand edit the XML.

Conventions (docs/conventions.md): x forward, y left, z up. Every joint axis is +y.
Positive hip and knee swing the foot backward. Positive wheel rotation rolls forward.
Left and right legs use identical signs.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.params import load_params

OUT_PATH = Path(__file__).resolve().parent / "generated" / "biped.xml"

BODY_PART = "2"
WHEEL = "4"
FLOOR_AFFINITY = "6"


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


def build_xml(p):
    validate(p)
    sim, b, th, sh, wh = p["sim"], p["body"], p["thigh"], p["shank"], p["wheel"]
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
            damping=fmt(jt["damping_Nms_per_rad"]), armature=fmt(wheel_armature),
        )
        ET.SubElement(
            wheel, "inertial", pos="0 0 0",
            mass=fmt(wh["mass_kg"]), diaginertia=fmt(*wh["inertia_kgm2"]),
        )
        ET.SubElement(
            wheel, "geom", name=f"wheel_{side}_geom", type="cylinder",
            fromto=fmt(0, -wh["width_m"] / 2, 0, 0, wh["width_m"] / 2, 0),
            size=fmt(wh["radius_m"]), friction=fmt(*ct["wheel_friction"]),
            contype=WHEEL, conaffinity="0", rgba="0.1 0.1 0.1 1",
        )

    act = ET.SubElement(root, "actuator")
    for side in ("l", "r"):
        for jn in ("hip", "knee"):
            lo, hi = jt[jn]["range_rad"]
            ET.SubElement(
                act, "position", name=f"{jn}_{side}_pos", joint=f"{jn}_{side}",
                kp=fmt(lm["kp_Nm_per_rad"]), kv=fmt(lm["kv_Nms_per_rad"]),
                ctrlrange=fmt(lo, hi),
                forcerange=fmt(-lm["peak_torque_Nm"], lm["peak_torque_Nm"]),
            )
        ET.SubElement(
            act, "motor", name=f"wheel_{side}_torque", joint=f"wheel_{side}", gear="1",
            ctrlrange=fmt(-wm["peak_torque_Nm"], wm["peak_torque_Nm"]),
        )

    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "framequat", name="orientation", objtype="site", objname="imu")
    ET.SubElement(sensors, "gyro", name="gyro", site="imu")
    ET.SubElement(sensors, "accelerometer", name="accel", site="imu")

    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def main():
    xml = build_xml(load_params())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(xml)
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
