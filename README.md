# EgoExo Forge: All the Data + Utilities Needed for Egocentric and Exocentric Human Data

Uses [Rerun](https://rerun.io/) to visualize, [Gradio](https://www.gradio.app) for an interactive UI, and [Pixi](https://pixi.sh/latest/) for a easy installation

<p align="center">
    <a title="Website" href="https://rerun.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/badge/Rerun-0.23.0-blue.svg?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzQ0MV8xMTAzOCkiPgo8cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHJ4PSI4IiBmaWxsPSJibGFjayIvPgo8cGF0aCBkPSJNMy41OTcwMSA1Ljg5NTM0TDkuNTQyOTEgMi41MjM1OUw4Ljg3ODg2IDIuMTQ3MDVMMi45MzMgNS41MTg3NUwyLjkzMjk1IDExLjI5TDMuNTk2NDIgMTEuNjY2MkwzLjU5NzAxIDUuODk1MzRaTTUuMDExMjkgNi42OTc1NEw5LjU0NTc1IDQuMTI2MDlMOS41NDU4NCA0Ljk3NzA3TDUuNzYxNDMgNy4xMjI5OVYxMi44OTM4SDcuMDg5MzZMNi40MjU1MSAxMi41MTczVjExLjY2Nkw4LjU5MDY4IDEyLjg5MzhIOS45MTc5NUw2LjQyNTQxIDEwLjkxMzNWMTAuMDYyMUwxMS40MTkyIDEyLjg5MzhIMTIuNzQ2M0wxMC41ODQ5IDExLjY2ODJMMTMuMDM4MyAxMC4yNzY3VjQuNTA1NTlMMTIuMzc0OCA0LjEyOTQ0TDEyLjM3NDMgOS45MDAyOEw5LjkyMDkyIDExLjI5MTVMOS4xNzA0IDEwLjg2NTlMMTEuNjI0IDkuNDc0NTRWMy43MDM2OUwxMC45NjAyIDMuMzI3MjRMMTAuOTYwMSA5LjA5ODA2TDguNTA2MyAxMC40ODk0TDcuNzU2MDEgMTAuMDY0TDEwLjIwOTggOC42NzI1MlYyLjk5NjU2TDQuMzQ3MjMgNi4zMjEwOUw0LjM0NzE3IDEyLjA5Mkw1LjAxMDk0IDEyLjQ2ODNMNS4wMTEyOSA2LjY5NzU0Wk05LjU0NTc5IDUuNzMzNDFMOS41NDU4NCA4LjI5MjA2TDcuMDg4ODYgOS42ODU2NEw2LjQyNTQxIDkuMzA5NDJWNy41MDM0QzYuNzkwMzIgNy4yOTY0OSA5LjU0NTg4IDUuNzI3MTQgOS41NDU3OSA1LjczMzQxWiIgZmlsbD0id2hpdGUiLz4KPC9nPgo8ZGVmcz4KPGNsaXBQYXRoIGlkPSJjbGlwMF80NDFfMTEwMzgiPgo8cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIGZpbGw9IndoaXRlIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==">
    </a>
    <a title="Personal Site" href="https://pablovela.dev/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-website.svg" alt="personal site">
    </a>
    <a title="Github" href="https://github.com/rerun-io/egoexo-forge" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/github/stars/rerun-io/egoexo-forge?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
    </a>
    <a title="Social" href="https://x.com/pablovelagomez1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
    </a>
</p>

<p align="center">
  <img src="media/egoexo-viewer.gif" alt="example output" width="720" />
</p>

## Installation
### Using Pixi
Make sure you have the [Pixi](https://pixi.sh/latest/#installation) package manager installed

This is Linux only with an NVIDIA GPU for batch conversion to RRD format

```bash
git clone https://github.com/rerun-io/egoexo-forge.git
cd egoexo-forge
pixi run app
```

All commands can be listed using `pixi task list`

## Usage
### Gradio App
```
pixi run app
```
### CLI (TODO)


You can see all tasks by running `pixi task list`

## Dataset Information

### Assembly101
A procedural activity dataset with 4321 multi-view videos of people assembling and disassembling 101 take-apart toy vehicles, featuring rich variations in action ordering, mistakes, and corrections.
**Note:** Assembly101 is very large and can take some time to fully load each episode.

### HoCap
A dataset for 3D reconstruction and pose tracking of hands and objects in videos, featuring humans interacting with objects for various tasks including pick-and-place actions and handovers.

### EgoDex
The largest and most diverse dataset of dexterous human manipulation with 829 hours of egocentric video and paired 3D hand tracking, covering 194 different tabletop tasks with everyday household objects.

### UmeTrack
Coming soon...

## Supported Datasets

### Currently Supported

- [Assembly101](https://assembly-101.github.io/) ✅
- [HO-Cap](https://irvlutd.github.io/HOCap/) ✅
- [EgoDex](https://arxiv.org/abs/2505.11709) ✅

### Planned Support

- UmeTrack 🚧

## Acknowledgements

### [Assembly101](https://assembly-101.github.io/)
```bibtex
@article{sener2022assembly101,
    title = {Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities},
    author = {F. Sener and D. Chatterjee and D. Shelepov and K. He and D. Singhania and R. Wang and A. Yao},
    journal = {CVPR 2022},
}
```

### [HO-Cap](https://irvlutd.github.io/HOCap/)
```bibtex
@misc{wang2024hocapcapturedataset3d,
      title={HO-Cap: A Capture System and Dataset for 3D Reconstruction and Pose Tracking of Hand-Object Interaction},
      author={Jikai Wang and Qifan Zhang and Yu-Wei Chao and Bowen Wen and Xiaohu Guo and Yu Xiang},
      year={2024},
      eprint={2406.06843},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.06843},
}
```

### [EgoDex](https://arxiv.org/abs/2505.11709)
```bibtex
@misc{hoque2025egodexlearningdexterousmanipulation,
      title={EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video},
      author={Ryan Hoque and Peide Huang and David J. Yoon and Mouli Sivapurapu and Jian Zhang},
      year={2025},
      eprint={2505.11709},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
