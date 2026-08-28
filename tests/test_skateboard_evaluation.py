import numpy as np

from sbto.evaluation.skateboard import (
    quaternion_angle_deg,
    quaternion_yaw_deg,
    torso_tilt_deg,
    trailing_true_duration,
)


def test_skateboard_orientation_and_contact_metrics():
    yaw = np.radians([170.0, 190.0, 350.0]) / 2.0
    quaternions = np.column_stack(
        [np.cos(yaw), np.zeros((3, 2)), np.sin(yaw)]
    )

    assert np.allclose(quaternion_yaw_deg(quaternions), [170.0, 190.0, 350.0])
    assert np.allclose(quaternion_angle_deg(quaternions, quaternions), 0.0)
    assert np.allclose(torso_tilt_deg(quaternions), 0.0)
    assert trailing_true_duration(np.array([True, False, True, True]), 0.01) == 0.02
