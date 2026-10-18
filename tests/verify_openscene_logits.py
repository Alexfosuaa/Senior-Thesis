import os, sys
import numpy as np
import torch

torch.Tensor.cuda = lambda self, *a, **k: self
import clip
_orig_clip_load = clip.load
def _cpu_clip_load(name, device='cpu', jit=False):
    return _orig_clip_load(name, device='cpu', jit=jit)
clip.load = _cpu_clip_load

sys.path.insert(0, os.path.join('external_src', 'segmentation'))
sys.path.insert(0, os.path.join('external_src', 'segmentation', 'openscene'))
sys.path.insert(0, os.path.join('external_src', 'segmentation', 'openscene', 'models'))

from openscene_model import OpenSceneModel
from util.util import extract_text_feature
from dataset.label_constants import SCANNET_LABELS_20
import MinkowskiEngine as ME

CONFIG_PATH = 'external_src/segmentation/openscene/config/scannet/ours_openseg.yaml'
SCENE_PTH = '/home/cpsc4900_afo6/scratch_cpsc4900/cpsc4900_afo6/openscene/scannet_3d/val/scene0011_00_vh_clean_2.pth'

device = torch.device('cpu')

model = OpenSceneModel(
    dataset_name='scannet',
    config_path=CONFIG_PATH,
    model_restore_path=None,
    device=device,
)

coords, colors, labels = torch.load(SCENE_PTH, map_location='cpu')
coords = np.asarray(coords)
n_points = coords.shape[0]
print('n_points (original):', n_points)

feats_t = torch.ones((n_points, 3), dtype=torch.float32)

voxel_size = model.config.voxel_size
coords_vox = np.floor(coords / voxel_size)
unique_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
    coords_vox, return_index=True, return_inverse=True
)
feats_unique = feats_t[unique_map]
coords_batched = ME.utils.batched_coordinates([unique_coords])
sinput = ME.SparseTensor(feats_unique, coordinates=coords_batched, device=device)

with torch.no_grad():
    point_features = model.forward(sinput)
print('point_features shape (unique voxels):', point_features.shape)

text_features = extract_text_feature(list(SCANNET_LABELS_20), model.config)

logits = point_features.float() @ text_features.float().t()
logits_per_point = logits[inverse_map]
preds_per_point = torch.max(logits_per_point, 1)[1]
print('logits_per_point shape:', logits_per_point.shape)
assert logits_per_point.shape[0] == n_points

# --- Sanity-check actual VALUES, not just shape ---
print('\n--- sample output check ---')
print('class order:', list(SCANNET_LABELS_20))
print('logits for point 0:', logits_per_point[0].tolist())
print('logits for point 1000:', logits_per_point[1000].tolist())
print('logits min/max/mean:', logits_per_point.min().item(), logits_per_point.max().item(), logits_per_point.mean().item())
print('any NaN:', torch.isnan(logits_per_point).any().item())
print('any Inf:', torch.isinf(logits_per_point).any().item())
print('predicted classes, first 20 points:', preds_per_point[:20].tolist())
print('number of distinct predicted classes across all points:', len(torch.unique(preds_per_point)))
probs_point0 = torch.softmax(logits_per_point[0], dim=0)
print('softmax probs for point 0:', probs_point0.tolist())
print('sum of probs for point 0 (should be ~1.0):', probs_point0.sum().item())