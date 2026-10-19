import os, sys
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

sys.path.insert(0, os.path.join('external_src', 'segmentation', 'mosaic3d'))

class Mosaic3DModel(object):
	'''
	Mosaic3D model wrapper.

	Arg(s):
		dataset_name : str
			dataset used
		cfg_file : str
			absolute path to Mosaic3D's `configs` directory
		model_restore_path : str
			path to pretrained model weights
		device : torch.device
			device to run model on
	'''
	MODEL_OVERRIDES = [
		'model=spunet34c+ppt',
		'model/clip=recap_clip',
		'model/loss=caption',
		'optim=sgd',
		'scheduler=onecyclelr',
		'+model.best_metric=${data.checkpoint_monitor}',
	]

	DATASET_TO_DATA_OVERRIDE = {
		'scannet': 'sc+ar+sc++',
	}

	def __init__(self,
	             dataset_name,
	             cfg_file,
	             model_restore_path=None,
	             device=torch.device('cuda')):

		self.dataset_name = dataset_name

		data_override = 'data={}'.format(
			self.DATASET_TO_DATA_OVERRIDE[dataset_name])

		with initialize_config_dir(config_dir=cfg_file, job_name='mosaic3d', version_base='1.3'):
			self.config = compose(
				config_name='eval',
				overrides=[data_override] + self.MODEL_OVERRIDES)

		self.model = instantiate(self.config.model)
		self.model.configure_model()

		self.device = device
		self.to(self.device)

		if model_restore_path is not None:
			self.restore_model(model_restore_path=model_restore_path)

		self.eval()

	def forward(self, batch_dict):
		'''
		Forwards input through network

		Arg(s):
			batch_dict : input
		Return(s):
			out_dict
				model output
		'''

		return self.model.forward(batch_dict)

	def parameters(self):
		'''
		Returns the list of parameters in the model

		Return(s):
			list[torch.Tensor[float32]] : list of parameters
		'''

		return list(self.model.parameters())

	def train(self):
		'''
		Sets model to training mode
		'''

		self.model.train()

	def eval(self):
		'''
		Sets model to evaluation mode
		'''

		self.model.eval()

	def to(self, device):
		'''
		Move model to a device

		Arg(s):
			device : torch.device
				device to use
		'''

		self.device = device
		self.model.to(device)

	def restore_model(self,
			  model_restore_path,
			  optimizer=None):
		'''
		Loads weight from checkpoint

		Arg(s):
			model_restore_path : str
				path to model weights
			optimizer : torch.optim
				optimizer to restore
		'''

		checkpoint = torch.load(model_restore_path, map_location='cpu')

		state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

		# for compatibility between ScanNet20 and ScanNet200 checkpoints
		for key in list(state_dict.keys()):
			if 'emb_target' in key:
				del state_dict[key]

		return self.model.load_state_dict(state_dict, strict=False)

	def save_model(self,
		       model_checkpoint_path,
		       step,
		       optimizer=None):
		'''
		Save weights of the model to checkpoint path

		Arg(s):
			model_checkpoint_path : str
				path to save checkpoint
			step : int
				current training step
			optimizer : torch.optim
				optimizer
		'''

		checkpoint = {
			'epoch': step,
			'state_dict': self.model.state_dict()
		}

		if optimizer is not None:
			checkpoint['optimizer'] = optimizer.state_dict()

		torch.save(
			checkpoint,
			model_checkpoint_path)