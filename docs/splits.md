# Dataset splits

## Dataset Split Description
For dataset split description in NAVSIM, please refer to [docs in the navsim repository](https://github.com/autonomousvision/navsim/blob/main/docs/splits.md).


## Our Designed Split

### Debug Split

We provide debug split `navtrain_debug` and `navtest_debug` for saving your time  in debugging. They are small subsets of `navtrain` and `navtest` respectively.

You can find the split in [navtrain_debug.yaml](../navsim/planning/script/config/common/train_test_split/scene_filter/navtrain_debug.yaml) and [navtest_debug.yaml](../navsim/planning/script/config/common/train_test_split/scene_filter/navtest_debug.yaml).

### Turning Scene Split

In our paper, we split `navtest` into 3 subsets according to the turning angle of gt trajectory: `navtest_l`, `navtest_f`, `navtest_r`. You can find the split in [navtest_angle_split_l.yaml](../navsim/planning/script/config/common/train_test_split/scene_filter/navtest_angle_split_l.yaml), [navtest_angle_split_f.yaml](../navsim/planning/script/config/common/train_test_split/scene_filter/navtest_angle_split_f.yaml), and [navtest_angle_split_r.yaml](../navsim/planning/script/config/common/train_test_split/scene_filter/navtest_angle_split_r.yaml).
