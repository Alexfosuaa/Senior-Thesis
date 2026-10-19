"""
Verify the Mosaic3D teacher wrapper on a real ScanNet validation scene.
"""

import os, sys
import torch
import hydra
from omegaconf import OmegaConf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, os.path.join(REPO_ROOT, 'src', 'segmentation'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'external_src', 'segmentation', 'mosaic3d'))

from mosaic3d_model import Mosaic3DModel
import src.utils.caption_utils as caption_utils
from src.data.metadata.scannet import CLASS_LABELS_20
from src.models.losses.clip_alignment_loss import CLIPAlignmentEval

CFG_DIR = os.path.join(REPO_ROOT, 'external_src', 'segmentation', 'mosaic3d', 'configs')
SCANNET_DATA_DIR = os.path.expanduser('~/scratch_cpsc4900/cpsc4900_afo6/mosaic3d/data/scannet')
CKPT_PATH = os.path.expanduser('~/scratch_cpsc4900/cpsc4900_afo6/mosaic3d/checkpoints/spunet34c.ckpt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[setup] device: {device}')

# Load model and pretrained checkpoint.
wrapper = Mosaic3DModel(
    dataset_name='scannet',
    cfg_file=CFG_DIR,
    model_restore_path=None,
    device=device,
)

restore_result = wrapper.restore_model(model_restore_path=CKPT_PATH)
print(f'[checkpoint] loaded: {CKPT_PATH}')
print(f'[checkpoint] missing_keys: {restore_result.missing_keys}')
print(f'[checkpoint] unexpected_keys: {restore_result.unexpected_keys}')
assert not restore_result.missing_keys, 'checkpoint load left missing_keys'
assert not restore_result.unexpected_keys, 'checkpoint load left unexpected_keys'

lit_model = wrapper.model.to(device)
lit_model.eval()

# Load a real ScanNet validation scene
OmegaConf.set_struct(wrapper.config, False)
wrapper.config.data.val_datasets[0].data_dir = SCANNET_DATA_DIR
print(f'[dataset] data_dir: {wrapper.config.data.val_datasets[0].data_dir}')
print(f'[dataset] target: {wrapper.config.data.val_datasets[0]._target_}')

val_dataset = hydra.utils.instantiate(wrapper.config.data.val_datasets[0])()
print(f'[dataset] size (num scenes): {len(val_dataset)}')

sample = val_dataset[0]
print(f'[dataset] sample keys: {list(sample.keys())}')

collate_fn = hydra.utils.instantiate(wrapper.config.data.collate_fn)
batch = collate_fn([sample])

def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    return x


batch = move_to_device(batch, device)

# Forward pass
with torch.no_grad():
    out_dict = wrapper.forward(batch)

clip_feat = out_dict['clip_feat']
print(f'[forward] clip_feat shape: {clip_feat.shape}')


# Score against CLIP text embeddings of the 20 ScanNet classes
class_names = list(CLASS_LABELS_20)
prompted_names = [f"a {c} in a scene" if "other" not in c else "other" for c in class_names]

clip_encoder = lit_model.clip_encoder.to(device)

hf_tokenizer = clip_encoder.text_tokenizer.tokenizer
context_length = clip_encoder.text_tokenizer.context_length

encoded = hf_tokenizer(
    prompted_names,
    return_tensors=None,
    max_length=context_length,
    padding='max_length',
    truncation=True,
)
text_tokens = torch.tensor(encoded['input_ids'], dtype=torch.long).to(device)

with torch.no_grad():
    text_features = clip_encoder.encode_text(text_tokens).float()
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
print(f'[text embeddings] shape: {text_features.shape}')

# Score using Mosaic3D's own CLIPAlignmentEval.
eval_module = CLIPAlignmentEval(normalize_input=True, text_clip_path=None)
eval_module.set_target_embedding(text_features)

with torch.no_grad():
    logits = eval_module.predict(clip_feat.float(), return_logit=True)
print(f'[logits] shape: {logits.shape}')


# Verification checks
assert logits.shape[0] == clip_feat.shape[0], 'logits/point count mismatch'
assert logits.shape[1] == len(class_names), 'logits/class count mismatch'

has_nan = torch.isnan(logits).any().item()
has_inf = torch.isinf(logits).any().item()
assert not has_nan, 'logits contain NaN'
assert not has_inf, 'logits contain Inf'

preds = torch.argmax(logits, dim=1)
num_distinct_classes = len(torch.unique(preds))

probs_point0 = torch.softmax(logits[0], dim=0)
prob_sum = probs_point0.sum().item()
assert abs(prob_sum - 1.0) < 1e-3, 'softmax does not sum to ~1.0'

print('\n--- sample output check ---')
print(f'class order: {class_names}')
print(f'logits for point 0: {logits[0].tolist()}')
print(f'logits for point 1000: {logits[min(1000, logits.shape[0]-1)].tolist()}')
print(f'logits min/max/mean: {logits.min().item()}, {logits.max().item()}, {logits.mean().item()}')
print(f'any NaN: {has_nan}')
print(f'any Inf: {has_inf}')
print(f'predicted classes, first 20 points: {preds[:20].tolist()}')
print(f'number of distinct predicted classes: {num_distinct_classes}')
print(f'softmax probs for point 0: {probs_point0.tolist()}')
print(f'sum of probs for point 0 (should be ~1.0): {prob_sum}')

print('\nALL CHECKS PASSED')