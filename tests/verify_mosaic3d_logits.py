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

CFG_DIR = os.path.join(REPO_ROOT, 'external_src', 'segmentation', 'mosaic3d', 'configs')
SCANNET_DATA_DIR = os.path.expanduser('~/scratch_cpsc4900/cpsc4900_afo6/mosaic3d/data/scannet')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device:', device)

wrapper = Mosaic3DModel(
    dataset_name='scannet',
    cfg_file=CFG_DIR,
    model_restore_path=None,
    device=device,
)

lit_model = wrapper.model.to(device)
lit_model.eval()

# Point the plain ScanNet20 val dataset at our actual extracted data,
# instead of the cluster-wide default path baked into the config.
OmegaConf.set_struct(wrapper.config, False)
wrapper.config.data.val_datasets[0].data_dir = SCANNET_DATA_DIR
print('using scannet data_dir:', wrapper.config.data.val_datasets[0].data_dir)
print('val dataset target:', wrapper.config.data.val_datasets[0]._target_)

# These configs use _partial_: true, so instantiate() returns a partial
# that needs to be called to actually build the object.
val_dataset = hydra.utils.instantiate(wrapper.config.data.val_datasets[0])()
print('val dataset size (num scenes):', len(val_dataset))

sample = val_dataset[0]
print('sample keys:', list(sample.keys()))

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

with torch.no_grad():
    out_dict = wrapper.forward(batch)

clip_feat = out_dict['clip_feat']
print('clip_feat shape:', clip_feat.shape)

class_names = list(CLASS_LABELS_20)
prompted_names = [f"a {c} in a scene" if "other" not in c else "other" for c in class_names]

clip_encoder = lit_model.clip_encoder.to(device)

# Work around this env's transformers/torch version mismatch, which breaks
# HFTokenizer.__call__'s hardcoded return_tensors='pt': call the underlying
# HF tokenizer ourselves with return_tensors=None and build the tensor by hand.
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
print('text_features shape:', text_features.shape)

point_features_norm = torch.nn.functional.normalize(clip_feat.float(), p=2, dim=1)
logits = torch.matmul(point_features_norm, text_features.float().t())
print('logits shape:', logits.shape)

assert logits.shape[0] == clip_feat.shape[0]
assert logits.shape[1] == len(class_names)

preds = torch.argmax(logits, dim=1)

print('\n--- sample output check ---')
print('class order:', class_names)
print('logits for point 0:', logits[0].tolist())
print('logits for point 1000:', logits[min(1000, logits.shape[0]-1)].tolist())
print('logits min/max/mean:', logits.min().item(), logits.max().item(), logits.mean().item())
print('any NaN:', torch.isnan(logits).any().item())
print('any Inf:', torch.isinf(logits).any().item())
print('predicted classes, first 20 points:', preds[:20].tolist())
print('number of distinct predicted classes:', len(torch.unique(preds)))
probs_point0 = torch.softmax(logits[0], dim=0)
print('softmax probs for point 0:', probs_point0.tolist())
print('sum of probs for point 0 (should be ~1.0):', probs_point0.sum().item())