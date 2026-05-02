"""Inline MuJoCo XML models and loader helpers."""
from __future__ import annotations

CARTPOLE_XML = """
<mujoco model="cartpole">
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <body name="cart">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".2 .1 .05" mass="1"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.04" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" gear="1"/>
  </actuator>
</mujoco>
"""

PENDULUM_XML = """
<mujoco model="pendulum">
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <body name="pole" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 1.0" size="0.04" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" gear="1"/>
  </actuator>
</mujoco>
"""

DOUBLE_PENDULUM_XML = """
<mujoco model="double_pendulum">
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="hinge1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.04" mass="1"/>
      <body name="link2" pos="0 0 0.5">
        <joint name="hinge2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.03" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge1" gear="1"/>
    <motor joint="hinge2" gear="1"/>
  </actuator>
</mujoco>
"""

MODELS = {
    "cartpole":         (CARTPOLE_XML,        "nq=2 nv=2 nu=1"),
    "pendulum":         (PENDULUM_XML,         "nq=1 nv=1 nu=1"),
    "double_pendulum":  (DOUBLE_PENDULUM_XML,  "nq=2 nv=2 nu=2"),
}
