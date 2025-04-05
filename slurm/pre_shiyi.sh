shiyi_workspace=/lustre/fsw/portfolios/av/users/shiyil

cd ${shiyi_workspace}/zxli/navsim_workspace/navsim2
git pull
pwd
source ${shiyi_workspace}/anaconda3/etc/profile.d/conda.sh
conda activate ${shiyi_workspace}/zxli/conda_navsim_v2

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${shiyi_workspace}/yaowenh/proj/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="${shiyi_workspace}/zxli/navsim_workspace/exp2"
export NAVSIM_DEVKIT_ROOT="${shiyi_workspace}/zxli/navsim_workspace/navsim2"
export OPENSCENE_DATA_ROOT="${shiyi_workspace}/yaowenh/proj/navsim_workspace/dataset"
export NAVSIM_TRAJPDM_ROOT="${shiyi_workspace}/yaowenh/proj/navsim_workspace/dataset/traj_pdm"
