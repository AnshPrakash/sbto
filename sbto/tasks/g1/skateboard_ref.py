from dataclasses import dataclass

import numpy as np

import sbto.tasks.g1.constants as G1
from sbto.sim.scene_mj import MjScene
from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.tasks.cost import (
    hamming_dist_nb,
    lower_bound_cost_nb,
    quadratic_cost_nb,
    quaternion_dist_logmap_nb,
)
from sbto.tasks.g1.robot_ref import ConfigG1RobotRef, G1RobotRef
from sbto.tasks.task_mj_ref import ConfigRefMotion


@dataclass
class ConfigG1SkateboardRef(ConfigG1RobotRef):
    board_pos_weight: float = 40.0
    board_quat_weight: float = 4.0
    board_quat_weight_terminal: float = 4.0
    board_v_weight: float = 0.2
    board_w_weight: float = 0.2
    deck_contact_weight: float = 2.0
    foot_board_position_weight: float = 20.0
    landing_duration: float = 0.3
    board_floor_contact_weight: float = 2.0
    foot_floor_collision_weight: float = 2.0
    robot_floor_collision_weight: float = 100.0
    torso_min_height: float = 0.65
    torso_min_height_weight: float = 10_000.0


class G1SkateboardRef(G1RobotRef):
    BOARD_FLOOR_SENSOR = "skateboard_floor"
    ROBOT_FLOOR_SENSOR = "robot_floor"
    DECK_CONTACT_SENSORS = ("left_foot_deck", "right_foot_deck")
    FOOT_BOARD_POSITION_SENSORS = (
        "left_foot_pos_board",
        "right_foot_pos_board",
    )
    FOOT_FLOOR_SENSORS = (
        "left_foot1_floor",
        "left_foot2_floor",
        "left_foot3_floor",
        "right_foot1_floor",
        "right_foot2_floor",
        "right_foot3_floor",
    )

    def __init__(
        self,
        sim: SimMjRollout,
        cfg: ConfigG1SkateboardRef,
        cfg_ref: ConfigRefMotion,
        mj_scene_ref: MjScene | None = None,
    ):
        super().__init__(sim, cfg, cfg_ref, mj_scene_ref)

        self.add_sensor_cost(
            G1.Sensors.TORSO_POS,
            lower_bound_cost_nb,
            sub_idx_sensor=2,
            ref_values=cfg.torso_min_height,
            weights=cfg.torso_min_height_weight,
        )

        self.add_state_cost_from_ref(
            "board_position",
            quadratic_cost_nb,
            sim.mj_scene.obj_pos_adr,
            weights=cfg.board_pos_weight,
            weights_terminal=cfg.board_pos_weight,
        )
        self.add_state_cost_from_ref(
            "board_quat",
            quaternion_dist_logmap_nb,
            sim.mj_scene.obj_quat_adr,
            weights=cfg.board_quat_weight,
            weights_terminal=cfg.board_quat_weight_terminal,
        )
        self.add_state_cost_from_ref(
            "board_velocity",
            quadratic_cost_nb,
            sim.mj_scene.obj_v_adr,
            weights=cfg.board_v_weight,
            weights_terminal=cfg.board_v_weight,
        )
        self.add_state_cost_from_ref(
            "board_angular_velocity",
            quadratic_cost_nb,
            sim.mj_scene.obj_w_adr,
            weights=cfg.board_w_weight,
            weights_terminal=cfg.board_w_weight,
        )

        self.ref.compute_sensor_data([
            *self.DECK_CONTACT_SENSORS,
            *self.FOOT_BOARD_POSITION_SENSORS,
            self.BOARD_FLOOR_SENSOR,
            self.ROBOT_FLOOR_SENSOR,
            *self.FOOT_FLOOR_SENSORS,
        ])
        self.add_sensor_cost_from_ref(
            self.FOOT_BOARD_POSITION_SENSORS,
            quadratic_cost_nb,
            weights=cfg.foot_board_position_weight,
            weights_terminal=cfg.foot_board_position_weight,
        )
        deck_contact_ref = np.column_stack([
            self.ref.sensor_data[name][:self.T, 0]
            for name in self.DECK_CONTACT_SENSORS
        ]).astype(bool).astype(np.int32)
        landing_steps = min(self.T, round(cfg.landing_duration / sim.mj_scene.dt))
        deck_contact_ref[-landing_steps:] = 1
        self.add_sensor_cost(
            self.DECK_CONTACT_SENSORS,
            hamming_dist_nb,
            ref_values=deck_contact_ref,
            weights=cfg.deck_contact_weight,
        )
        board_floor_ref = self.ref.sensor_data[self.BOARD_FLOOR_SENSOR][:self.T]
        self.add_sensor_cost(
            self.BOARD_FLOOR_SENSOR,
            hamming_dist_nb,
            ref_values=board_floor_ref,
            weights=cfg.board_floor_contact_weight,
        )
        self.add_sensor_cost(
            self.FOOT_FLOOR_SENSORS,
            hamming_dist_nb,
            ref_values=np.zeros((self.T, len(self.FOOT_FLOOR_SENSORS))),
            weights=cfg.foot_floor_collision_weight,
        )
        self.add_sensor_cost(
            self.ROBOT_FLOOR_SENSOR,
            hamming_dist_nb,
            ref_values=np.zeros((self.T, 1)),
            weights=cfg.robot_floor_collision_weight,
        )
        self.contact_plan = deck_contact_ref
        self.contact_obs_id = self.get_sensors_adr(self.DECK_CONTACT_SENSORS)
