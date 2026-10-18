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
├── tests/
│ └── verify_openscene_logits.py # Confirms a wrapper’s forward pass, combined
│ # with text-feature scoring, produces correctly
│ # shaped per-point logits on real ScanNet data
└── .gitignore


The `external_src/segmentation/*` folders are git submodules pinned to the
exact commit of each original codebase this project was built and verified
against. Each wrapper in `src/segmentation/` loads the corresponding
submodule and exposes a consistent interface (primarily `forward()`) for
use in the distillation pipeline.

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

Each teacher model uses its own conda environment on the HPC cluster
(environments kept outside version control under `ycrc_conda/`, see
`.gitignore`):

```bash
module load miniconda
conda activate openscene   # or: pla, mosaic3d
```

### Dataset

ScanNet access requires agreeing to the official
[Terms of Use](http://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf) and
emailing it to scannet@googlegroups.com. Each teacher model expects its
own specific preprocessed format:

- **OpenScene**: preprocessed release downloaded via
  `scripts/download_dataset.sh` in the OpenScene submodule, split into
  `scannet_3d/{train,val}` (`.pth` files: coords, colors, labels) and
  `scannet_2d/<scene>/` (fused 2D features). Downloaded and verified.
- **PLA**: preprocessed `.pth` files + caption/text-embedding data; see
  `external_src/segmentation/pla/docs/DATASET.md`. Not yet downloaded.
- **Mosaic3D**: geometry from
  [Pointcept/scannet-compressed](https://huggingface.co/datasets/Pointcept/scannet-compressed)
  combined with annotation files from
  [Mosaic3D on Hugging Face](https://huggingface.co/datasets/junhalee/Mosaic3D).
  Not yet downloaded.

## Usage

Example: running the OpenScene wrapper's forward pass on a loaded point
cloud (see `tests/verify_openscene_logits.py` for the full worked example,
including voxelization and converting the output into per-class logits):

```python
import sys, os
sys.path.insert(0, 'src/segmentation')
from openscene_model import OpenSceneModel

model = OpenSceneModel(
    dataset_name='scannet',
    config_path='external_src/segmentation/openscene/config/scannet/ours_openseg.yaml',
    model_restore_path=None,  # or path to a pretrained checkpoint
)

# sinput: a MinkowskiEngine.SparseTensor built from voxelized point coordinates
point_features = model.forward(sinput)  # -> [N_voxels, 768] per-point embeddings
```

## Status

- [x] OpenScene wrapper
- [x] PLA wrapper
- [x] Mosaic3D wrapper
- [x] OpenScene dataset downloaded and verified
- [ ] PLA dataset downloaded and verified
- [ ] Mosaic3D dataset downloaded and verified
- [x] OpenScene output verification (wrapper's forward pass + text-feature scoring
      produces correctly shaped, per-point logits on real ScanNet data — see
      `tests/verify_openscene_logits.py`)
- [ ] PLA output verification
- [ ] Mosaic3D output verification