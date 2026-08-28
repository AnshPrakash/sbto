import numpy as np

from sbto.sim.scene_mj import ConfigMjScene, MjScene
from sbto.utils.extract_ref import ReferenceMotion

MODEL = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="robot">
      <freejoint/>
      <geom size="0.1"/>
      <body>
        <joint name="actuated" type="hinge"/>
        <geom size="0.1"/>
      </body>
    </body>
    <body name="object">
      <freejoint/>
      <geom size="0.1"/>
      <body>
        <joint name="passive" type="hinge"/>
        <geom size="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="actuated"/>
  </actuator>
</mujoco>
"""


def test_reference_keeps_constant_passive_joint_and_final_frame(tmp_path):
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    scene = MjScene(ConfigMjScene(xml_scene_path=str(model_path)))
    qpos = np.tile(scene.mj_model.qpos0, (4, 1))
    passive_qpos = int(scene.mj_model.joint("passive").qposadr[0])
    qpos[:, passive_qpos] = 0.25
    qpos[:, scene.act_qposadr[0]] = [0.0, 0.1, 0.2, 0.3]
    ref_path = tmp_path / "reference.npz"
    np.savez(ref_path, qpos=qpos, fps=50.0)

    ref = ReferenceMotion(scene, str(ref_path), flip_quat_pos=False)

    assert ref.x.shape == (7, scene.Nx)
    assert ref.time[-1] == 0.06
    assert np.all(ref.x[:, passive_qpos] == 0.25)
    passive_velocity = scene.Nq + int(scene.mj_model.joint("passive").dofadr[0])
    assert np.all(ref.x[:, passive_velocity] == 0.0)


def test_reference_rejects_moving_passive_joints(tmp_path):
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    scene = MjScene(ConfigMjScene(xml_scene_path=str(model_path)))
    qpos = np.tile(scene.mj_model.qpos0, (4, 1))
    passive_qpos = int(scene.mj_model.joint("passive").qposadr[0])
    qpos[:, passive_qpos] = [0.0, 0.1, 0.2, 0.3]
    ref_path = tmp_path / "reference.npz"
    np.savez(ref_path, qpos=qpos, fps=50.0)

    try:
        ReferenceMotion(scene, str(ref_path), flip_quat_pos=False)
    except ValueError as exc:
        assert "moving passive joints" in str(exc)
    else:
        raise AssertionError("moving passive joint reference was accepted")


def test_reference_preserves_model_ordered_qvel_and_moving_passive_joint(tmp_path):
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    scene = MjScene(ConfigMjScene(xml_scene_path=str(model_path)))
    qpos = np.tile(scene.mj_model.qpos0, (4, 1))
    qvel = np.zeros((4, scene.Nv))
    passive_joint = scene.mj_model.joint("passive")
    passive_qpos = int(passive_joint.qposadr[0])
    passive_qvel = int(passive_joint.dofadr[0])
    qpos[:, passive_qpos] = [0.0, 0.1, 0.2, 0.3]
    qvel[:, passive_qvel] = 5.0
    ref_path = tmp_path / "reference.npz"
    np.savez(ref_path, qpos=qpos, qvel=qvel, fps=50.0)

    ref = ReferenceMotion(scene, str(ref_path), flip_quat_pos=False)

    assert ref.x.shape == (7, scene.Nx)
    np.testing.assert_allclose(ref.x[::2, passive_qpos], qpos[:, passive_qpos])
    np.testing.assert_allclose(
        ref.x[:, scene.Nq + passive_qvel], qvel[0, passive_qvel]
    )
