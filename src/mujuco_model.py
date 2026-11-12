import mujoco

def build_xml(p, mode="constraint", timestep=0.0005, friction=1.5, solref="0.001 1", integrator="implicitfast", solimp="0.99 0.999 0.0001"):
    """MJCF for the planar wheeled pendulum, built from params.yaml values.
    mode "constraint": rolling without slipping enforced by an equality constraint (x = r * wheel angle).
    mode "contact":    a real wheel rolling on a floor with friction (needs tuning, see Step 11)."""
    r, Mw = p.wheel_radius_m, p.wheel_mass_total_kg
    m, l, I = p.body_mass_kg, p.axle_to_com_m, p.body_pitch_inertia_kgm2
    g = p.gravity_ms2
    if mode == "constraint":
        wheel_geom = f'<geom type="cylinder" size="{r} 0.01" euler="1.5707963 0 0" mass="{Mw}" contype="0" conaffinity="0"/>'
        slides = '<joint name="slide_x" type="slide" axis="1 0 0"/>'
        floor = ""
        equality = f'<equality><joint joint1="slide_x" joint2="wheel_rot" polycoef="0 {r} 0 0 0" solref="{solref}" solimp="{solimp}"/></equality>'
    else:
        wheel_geom = (f'<geom type="cylinder" size="{r} 0.01" euler="1.5707963 0 0" mass="{Mw}" '
                      f'friction="{friction} 0.005 0.0001" solref="{solref}" solimp="{solimp}" condim="3"/>')
        slides = ('<joint name="slide_x" type="slide" axis="1 0 0"/>'
                  '<joint name="slide_z" type="slide" axis="0 0 1"/>')
        floor = f'<geom name="floor" type="plane" size="20 1 0.1" friction="{friction} 0.005 0.0001" solref="{solref}" solimp="{solimp}"/>'
        equality = ""
    return f"""
<mujoco model="balancer">
  <compiler angle="radian"/>
  <option timestep="{timestep}" integrator="{integrator}" gravity="0 0 {-g}"/>
  <worldbody>
    {floor}
    <body name="cart" pos="0 0 {r}">
      {slides}
      <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>
      <body name="wheel">
        <joint name="wheel_rot" type="hinge" axis="0 1 0"/>
        {wheel_geom}
      </body>
      <body name="body">
        <joint name="pitch" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 {l}" mass="{m}" diaginertia="0.004 {I} 0.004"/>
      </body>
    </body>
  </worldbody>
  {equality}
  <actuator>
    <motor name="wheel_motor" joint="wheel_rot" gear="1"/>
    <motor name="body_motor" joint="pitch" gear="1"/>
  </actuator>
</mujoco>
"""

def make(p, **kw):
    model = mujoco.MjModel.from_xml_string(build_xml(p, **kw))
    data = mujoco.MjData(model)
    ids = {"x": model.joint("slide_x").qposadr[0], "wheel": model.joint("wheel_rot").qposadr[0],
           "pitch": model.joint("pitch").qposadr[0]}
    vids = {"x": model.joint("slide_x").dofadr[0], "wheel": model.joint("wheel_rot").dofadr[0],
            "pitch": model.joint("pitch").dofadr[0]}
    return model, data, ids, vids

def apply_torque(data, tau):
    """Torque tau on the wheel, reaction -tau on the body (matches the derivation)."""
    data.ctrl[0] = tau
    data.ctrl[1] = -tau