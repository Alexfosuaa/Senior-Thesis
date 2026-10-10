# Point Cloud Semantic Segmentation via Knowledge Distillation

CS Senior Thesis — Yale University

Author: Alexandra Owusu

Advisor: Alex Wong

Course: CPSC 4900

## Overview

This project builds a smaller ("student") point cloud semantic segmentation
model via knowledge distillation from larger, open-vocabulary 3D scene
understanding models ("teachers"). Performance is evaluated using mIoU.

Teacher models currently integrated:
- [OpenScene](https://github.com/pengsongyou/openscene)
- [PLA](https://github.com/CVMI-Lab/PLA)
- [Mosaic3D](https://github.com/NVlabs/Mosaic3D)

Dataset: [ScanNet](http://www.scan-net.org/) (official access obtained).

## Repository Structure
```
├── src/
│ └── segmentation/ # Wrapper code for each teacher model
│ ├── openscene_model.py
│ ├── pla_model.py
│ ├── mosaic3d_model.py
│ └── template_model.py
├── external_src/
│ └── segmentation/ # Original teacher codebases (git submodules)
│ ├── openscene/
│ ├── pla/
│ └── mosaic3d/
└── .gitignore
```

The `external_src/segmentation/*` folders are git submodules pinned to the
exact commit of each original codebase this project was built and verified
against. Each wrapper in `src/segmentation/` loads the corresponding
submodule and exposes a consistent interface for use in the distillation
pipeline.

## Setup

### Clone with submodules

```bash
git clone --recurse-submodules https://github.com/Alexfosuaa/Senior-Thesis
```

If you already cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

### Environment

<!-- TODO: fill in your actual conda/environment setup, e.g.
module load miniconda
conda env create -f environment.yml
conda activate <env-name>
-->

### Dataset

ScanNet access requires agreeing to the official
[Terms of Use](http://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf) and
emailing it to scannet@googlegroups.com. Each teacher model expects its
own specific preprocessed format:

- **OpenScene**: preprocessed release via `scripts/download_dataset.sh`
  in the OpenScene submodule.
- **PLA**: preprocessed `.pth` files + caption/text-embedding data; see
  `external_src/segmentation/pla/docs/DATASET.md`.
- **Mosaic3D**: geometry from
  [Pointcept/scannet-compressed](https://huggingface.co/datasets/Pointcept/scannet-compressed)
  combined with annotation files from
  [Mosaic3D on Hugging Face](https://huggingface.co/datasets/junhalee/Mosaic3D).

<!-- TODO: add exact download/preprocessing commands once run -->

## Usage

<!-- TODO: add how to invoke each wrapper, e.g.
from src.segmentation.openscene_model import OpenSceneModel
model = OpenSceneModel(...)
-->

## Status

- [x] OpenScene wrapper 
- [x] PLA wrapper 
- [x] Mosaic3D wrapper 
- [ ] Dataset download and preprocessing per teacher
- [ ] Output verification (confirming wrapper outputs match original codebases)
