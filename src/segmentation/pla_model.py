import os, sys
import torch
import logging

sys.path.insert(0, os.path.join('external_src', 'segmentation'))
sys.path.insert(0, os.path.join('external_src', 'segmentation', 'pla'))
sys.path.insert(0, os.path.join('external_src', 'segmentation', 'pla', 'pcseg'))

from pcseg.models import build_vision_network
import pcseg.config as config

class _DatasetStub(object):
	'''
	Minimal stand-in for a PLA dataset object

	Arg(s):
		class_names : list[str]
			list of class names
		dataset_cfg : easydict.EasyDict
			dataset configuration section of PLA config
	'''

	def __init__(self, class_names, dataset_cfg):
		self.class_names = class_names
		self.dataset_cfg = dataset_cfg
		self.ignore_label = dataset_cfg.IGNORE_LABEL
		self.valid_class_idx = list(range(len(class_names)))

		if 'ignore_class_idx' in dataset_cfg:
			for c in dataset_cfg.ignore_class_idx:
				self.valid_class_idx.remove(c)

class PLAModel(object):
	'''
	PLA model wrapper.

	Arg(s):
		dataset_name : str
			dataset used
		cfg_file : str
			path to PLA configuration (yaml) file
		model_restore_path : str
			path to pretrained model weights
		device : torch.device
			device to run model on
	'''
	def __init__(self,
	             dataset_name,
	             cfg_file,
	             model_restore_path=None,
	             device=torch.device('cuda')):

		self.dataset_name = dataset_name

		self.config = config.cfg_from_yaml_file(cfg_file, config.cfg)

		dataset_stub = _DatasetStub(
			class_names=self.config.CLASS_NAMES,
			dataset_cfg=self.config.DATA_CONFIG)

		self.model = build_vision_network(
			model_cfg=self.config.MODEL,
			num_class=len(self.config.CLASS_NAMES),
			dataset=dataset_stub)

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
			ret_dict
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

		logger = logging.getLogger(__name__)

		return self.model.load_params_from_file(
			filename=model_restore_path,
			logger=logger,
			to_cpu=(self.device.type == 'cpu'))

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