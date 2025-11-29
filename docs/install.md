# Download and installation

To get started with NAVSIM:

### 1. Clone the repository

```bash
git clone https://github.com/William-Yao-2000/DriveSuprim.git
cd DriveSuprim
```

### 2. Download the dataset

You need to download the OpenScene logs and sensor blobs, as well as the nuPlan maps.
We provide scripts to download the nuplan maps, the mini split and the test split.
Navigate to the download directory and download the maps

**NOTE: Please check the [LICENSE file](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) before downloading the data.**

```bash
cd download && ./download_maps
```

Next download the data splits you want to use.
Note that the dataset splits do not exactly map to the recommended standardized training / test splits-
Please refer to [splits](splits.md) for an overview on the standardized training and test splits including their size and check which dataset splits you need to download in order to be able to run them.
You can download these splits with the following scripts.

```bash
./download_mini
./download_trainval
./download_test
./download_warmup_two_stage
./download_navhard_two_stage
./download_private_test_hard_two_stage
```

Also, the script `./download_navtrain` can be used to download a small portion of the  `trainval` dataset split which is needed for the `navtrain` training split.

This will download the splits into the download directory. From there, move it to create the following structure.
(Our paper only uses the `trainval` and `test` splits, so you can just download them.)

```angular2html
~/drivesuprim_workspace
├── DriveSuprim (containing the devkit)
├── exp_v2
└── dataset
    ├── maps
    ├── navsim_logs
    |    ├── test
    |    ├── trainval
    |    ├── private_test_hard
    |    |         └── private_test_hard.pkl
    │    └── mini
    └── sensor_blobs
    |    ├── test
    |    ├── trainval
    |    ├── private_test_hard
    |    |         ├──  CAM_B0
    |    |         ├──  CAM_F0
    |    |         ├──   ...
    |    └── mini
    └── navhard_two_stage
    |    ├── openscene_meta_datas
    |    ├── sensor_blobs
    |    ├── synthetic_scene_pickles
    |    └── synthetic_scenes_attributes.csv
    └── warmup_two_stage
    |    ├── openscene_meta_datas
    |    ├── sensor_blobs
    |    ├── synthetic_scene_pickles
    |    └── synthetic_scenes_attributes.csv
    └── private_test_hard_two_stage
    |    ├── openscene_meta_datas
    |    └── sensor_blobs
    └── traj_pdm_v2
```
Set the required environment variables.\
Based on the structure above, the environment variables need to be defined as:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/drivesuprim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/drivesuprim_workspace/exp_v2"
export NAVSIM_DEVKIT_ROOT="$HOME/drivesuprim_workspace/DriveSuprim"
export OPENSCENE_DATA_ROOT="$HOME/drivesuprim_workspace/dataset"
```

### 3. Create directory for storing PDM scores in the trajectory vocabulary

```bash
export NAVSIM_TRAJPDM_ROOT="$HOME/drivesuprim_workspace/dataset/traj_pdm_v2"
mkdir $NAVSIM_TRAJPDM_ROOT
```

### 3. Install the navsim-devkit

Finally, install navsim.
To this end, create a new environment and install the required dependencies:

```bash
conda env create --name drivesuprim -f environment.yml
conda activate drivesuprim
pip install -e .
```
