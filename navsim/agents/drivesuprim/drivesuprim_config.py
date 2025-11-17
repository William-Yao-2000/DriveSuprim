from dataclasses import dataclass
from typing import Tuple

from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.transfuser.transfuser_config import TransfuserConfig


@dataclass
class InferConfig:
    model: str = "teacher"  # teacher or student
    use_first_stage_traj_in_infer: bool = False
    save_pickle: bool = False


@dataclass
class EgoPerturbConfig:
    n_student_rotation_ensemble: int = 3
    offline_aug_angle_boundary: float = 0
    offline_aug_file: str = '???'


@dataclass
class RefinementConfig:
    use_multi_stage: bool = False
    refinement_approach: str = "transformer_decoder"
    num_refinement_stage: int = 1  # 2
    stage_layers: str = "3"  # "3" or "3+3" ...
    topks: str = "256"  # "256" or "256+64" ...
    use_mid_output: bool = True
    use_separate_stage_heads: bool = True
    use_imi_learning_in_refinement: bool = True


@dataclass
class DriveSuprimConfig(TransfuserConfig):
    ckpt_path: str = None

    seq_len: int = 2
    trajectory_imi_weight: float = 1.0
    trajectory_pdm_weight = {
        'no_at_fault_collisions': 3.0,
        'drivable_area_compliance': 3.0,
        'time_to_collision_within_bound': 4.0,
        'ego_progress': 2.0,
        'driving_direction_compliance': 1.0,
        'lane_keeping': 2.0,
        'traffic_light_compliance': 3.0,
        'history_comfort': 1.0,
    }

    vocab_size: int = 8192
    vocab_path: str = None
    normalize_vocab_pos: bool = False
    
    num_ego_status: int = 1
    sigma: float = 0.5
    vadv2_head_nhead: int = 8
    vadv2_head_nlayers: int = 3

    trajectory_sampling: TrajectorySampling = TrajectorySampling(
        time_horizon=4, interval_length=0.1
    )

    # img backbone
    backbone_type: str = 'resnet34'
    vit_ckpt: str = ''
    intern_ckpt: str = ''
    vov_ckpt: str = ''
    swin_ckpt: str = ''
    sptr_ckpt: str = ''
    
    lr_mult_backbone: float = 1.0
    backbone_wd: float = 0.0

    n_camera: int = 3  # 1 or 3 or 5

    camera_width: int = 2048
    camera_height: int = 512
    img_vert_anchors: int = camera_height // 32
    img_horz_anchors: int = camera_width // 32

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    training: bool = True

    # Augmentation setting
    only_ori_input: bool = False  # 如果是 True，说明是原来的训练设置
    ego_perturb: EgoPerturbConfig = EgoPerturbConfig()
    ori_vocab_pdm_score_full_path: str = "???"
    aug_vocab_pdm_score_dir: str = "???"

    # Self-distillation
    soft_label_traj: str = 'first'  # first or final
    soft_label_imi_diff_thresh: float = 1.0
    soft_label_score_diff_thresh: float = 0.15
    update_buffer_in_ema: bool = False

    refinement: RefinementConfig = RefinementConfig()
    inference: InferConfig = InferConfig()
