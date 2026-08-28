# Sampling Based Trajectory Optimization (SBTO)

This repository contains the official implementation of the SBTO mentioned in the paper:

> **DynaRetarget: Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization**
> Victor Dhedin, Ilyass Taouil, Shafeef Omar, Dian Yu, Kun Tao, Angela Dai, Majid Khadiv
> arXiv:2602.06827 · [Paper](https://arxiv.org/abs/2602.06827)

DynaRetarget is a complete pipeline for retargeting human motions to humanoid control policies. The core component is a novel Sampling-Based Trajectory Optimization (SBTO) framework that refines imperfect kinematic trajectories into dynamically feasible motions. SBTO incrementally advances the optimization horizon, enabling optimization over the entire trajectory for long-horizon tasks. The framework generalizes across varying object properties such as mass, size, and geometry using the same tracking objective.

## Dependencies

- Python 3.12.11
- MuJoCo, NumPy, Numba, SciPy, Hydra, OpenCV, and plotting/data utilities
- Exact resolved versions are recorded in `uv.lock`; requirements live in
  `pyproject.toml`

### Install
#### Environment
```bash
git clone https://github.com/AnshPrakash/sbto.git
cd sbto
uv sync --extra dev
uv run pytest -q tests
```

This project uses [Astral `uv`](https://docs.astral.sh/uv/) and the committed
`uv.lock` for a reproducible Python 3.12 environment. Prefix project commands
with `uv run`; no Conda activation is required.

#### OmniRetarget
Download robot-object motion references from Omniretarget dataset.
```bash
mkdir datasets && cd datasets
wget "https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset/resolve/main/robot-object.zip"
unzip robot-object.zip
```

## Usage
Most SBTO parameters can be set at runtime as command-line arguments. The code
base relies on [Hydra](https://hydra.cc/) to do so. Parameters required to
instantiate the different classes can be found in the `sbto/conf` directories.

### Loading a motion reference
To run SBTO on a specific motion reference from the OmniRetarget dataset simply run:
```bash
uv run python sbto/main.py \
  solver=cem \
  task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz
```
One can have more control on the motion reference by changing the parameters defined in the respective config [file](sbto/conf/task/g1/cfg_ref/default.yaml).

**Warning**: If you use your own reference motion in MuJoCo format then you should set `task.cfg_ref.flip_quat_pos=False`. This is set to True by default as for OmniRetarget data, free joints are expressed in [quat, pos] format.

To check that your reference is being loaded correctly, you can visualize it by running:
```bash
uv run python scripts/visualize_ref.py \
  task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz \
  task.cfg_ref.speedup=2.
```

### Articulated G1 skateboard reference

The skateboard task expects a numeric `.npz` with `qpos` shaped `(T, 49)`, a
scalar `fps`, and optionally model-ordered `qvel` shaped `(T, 47)`. `qpos`
follows the bundled MuJoCo order: G1 floating base and 29 joints, skateboard
free joint, then six passive truck/wheel joints. Provide `qvel` whenever a
passive joint moves; pose-only files remain supported when passive joints are
constant. MuJoCo free-joint poses use `[position, quaternion]` with a `wxyz`
quaternion, so disable the OmniRetarget conversion.

Convert a trusted BeyondMimic archive by joint name instead of assuming its
column order:

```bash
uv run --locked python scripts/convert_beyondmimic_reference.py \
  PATH_TO_BEYONDMIMIC.npz \
  artifacts/reference-sbto-state.npz
```

Then optimize it:

```bash
uv run python sbto/main.py \
  task=g1/skateboard_ref \
  task.cfg_ref.motion_path=PATH_TO_REFERENCE.npz \
  task.cfg_ref.flip_quat_pos=false \
  task.sim.cfg.step_knots=10
```

SBTO still optimizes only the 29 robot actuator targets. Passive skateboard
positions and velocities are reference state, not controls. See [ADR
0001](docs/adr/0001-articulated-skateboard-refinement.md) for the model and
validation boundary. Generated trajectories should stay outside Git until
their tracking, board rotation, landing, collision, and finite-value checks
pass.

### Changing the scene
SBTO also allows to change the scene directly from command line arguments.

Very importantly, SBTO loads **two different scenes** when using a reference: the one of the demonstration and the one of the refinement process (in which the rollouts happen).

Predefined scenes are already defined [here](sbto/conf/task/g1/mj_scene_ref) (for the reference) and [here](sbto/conf/task/g1/sim/mj_scene) (for the rollouts).

For the OmniRetarget dataset, the reference is a box. For the rollouts one can use different options with different objects:
```bash
uv run python sbto/main.py \
  solver=cem \
  task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz \
  task/g1/sim/mj_scene@task.sim.mj_scene=small_box
```

If you want to add your own objects, SBTO supports primitive geometries, `.urdf` and `.obj` meshes. Note the object placement has to be manually refined so that it starts in the correct position and orientation.

If you want to visualize your scene, you can change the reference's scene and use the same script as before:
```bash
uv run python scripts/visualize_ref.py \
  task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz \
  task/g1/mj_scene_ref@task.mj_scene_ref=../sim/mj_scene/chair_mesh
```

#### Without object
If you don't have any object in your scene use `g1/robot_ref` task:
```bash
uv run python sbto/main.py \
  task=g1/robot_ref \
  task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz
```

## Citation
If you use this code in your research, please cite:
```bibtex
@article{dhedin2025dynaretarget,
  title     = {DynaRetarget: Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization},
  author    = {Dhedin, Victor and Taouil, Ilyass and Omar, Shafeef and Yu, Dian and Tao, Kun and Dai, Angela and Khadiv, Majid},
  journal   = {arXiv preprint arXiv:2602.06827},
  year      = {2025}
}
```
