<div id="top" align="center">

<p align="center">
  <h2 align="left">DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning</h1>
  <h3 align="left"><a href="https://arxiv.org/abs/2506.06659">Paper Link</a>
</p>

</div>

<!-- TODO: check main page -->

<br/>

> [**DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning**](https://arxiv.org/abs/2406.15349)
>
> [Wenhao Yao](https://william-yao-2000.github.io/)<sup>1,2</sup>, [Zhenxin Li](https://woxihuanjiangguo.github.io/)<sup>1,2</sup>, [Shiyi Lan](https://voidrank.github.io/)<sup>3</sup>, [Zi Wang](https://scholar.google.com/citations?user=0SuL2yUAAAAJ&hl=en)<sup>3</sup>, [Xinglong Sun](https://www.xinglongsun.com/)<sup>3</sup>, [Jose M. Alvarez](https://alvarezlopezjosem.github.io/)<sup>3</sup>, [Zuxuan Wu](https://zxwu.azurewebsites.net/)<sup>1,2</sup>  <br>
>
> <sup>1</sup>Shanghai Key Lab of Intell. Info. Processing, School of CS, Fudan University \
> <sup>2</sup>Shanghai Collaborative Innovation Center of Intelligent Visual Computing \
> <sup>3</sup>NVIDIA
>
> Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2026
<br/>


## Getting started <a name="gettingstarted"></a>

- [Download and Installation](docs/install.md)
- [Dataset Splits](docs/splits.md)
- [Trajectory Score Generation](docs/score_generation.md)
- [Augmented Data Generation](docs/augmentation.md)
- [Training and Evaluation](docs/train_eval.md)

## Model Checkpoints

|Model|Resolution|Backbone | EPDMS | Checkpoint |
|:-|:-:|:---:|:----:|:----:|
|[DriveSuprim_R34](navsim/planning/script/config/common/agent/drivesuprim_agent_r34.yaml) | 512x2048 | ResNet34 | 83.1 | [link](https://huggingface.co/alkaid-2000/DriveSuprim/resolve/main/model_ckpt/drivesuprim_r34.ckpt) |
|[DriveSuprim_V2-99](navsim/planning/script/config/common/agent/drivesuprim_agent_vov.yaml) | 512x2048 | [V2-99](https://huggingface.co/alkaid-2000/DriveSuprim/resolve/main/pretrained_backbones/dd3d_det_final.pth) | 86.0 | [link](https://huggingface.co/alkaid-2000/DriveSuprim/resolve/main/model_ckpt/drivesuprim_vov.ckpt) |
|[DriveSuprim_ViT-L](navsim/planning/script/config/common/agent/drivesuprim_agent_vit.yaml) | 256x1024 | [ViT-Large](https://huggingface.co/alkaid-2000/DriveSuprim/resolve/main/pretrained_backbones/da_vitl16.pth) | 87.1 | [link](https://huggingface.co/alkaid-2000/DriveSuprim/resolve/main/model_ckpt/drivesuprim_vit.ckpt) |


## License and citation <a name="licenseandcitation"></a>

All assets and code in this repository are under the [Apache 2.0 license](./LICENSE) unless specified otherwise. The datasets (including nuPlan and OpenScene) inherit their own distribution licenses. Please consider citing our paper and project if they help your research.

```BibTeX
@inproceedings{Dauner2024NEURIPS,
	author = {Daniel Dauner and Marcel Hallgarten and Tianyu Li and Xinshuo Weng and Zhiyu Huang and Zetong Yang and Hongyang Li and Igor Gilitschenski and Boris Ivanovic and Marco Pavone and Andreas Geiger and Kashyap Chitta},
	title = {NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking},
	booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
	year = {2024},
}
```

```BibTeX
@misc{Contributors2024navsim,
    title={NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking},
    author={NAVSIM Contributors},
    howpublished={\url{https://github.com/autonomousvision/navsim}},
    year={2024}
}
```


## Other resources <a name="otherresources"></a>

- [SLEDGE](https://github.com/autonomousvision/sledge) | [tuPlan garage](https://github.com/autonomousvision/tuplan_garage) | [CARLA garage](https://github.com/autonomousvision/carla_garage) | [Survey on E2EAD](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)
- [PlanT](https://github.com/autonomousvision/plant) | [KING](https://github.com/autonomousvision/king) | [TransFuser](https://github.com/autonomousvision/transfuser) | [NEAT](https://github.com/autonomousvision/neat)

