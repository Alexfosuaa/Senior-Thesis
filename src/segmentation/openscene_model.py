import os, sys
import urllib.parse
import torch
from torch.utils import model_zoo

sys.path.insert(0, os.path.join('external_src', 'segmentation'))
sys.path.insert(0, os.path.join('external_src', 'segmentation', 'openscene'))
sys.path.insert(0, os.path.join('external_src', 'segmentation', 'openscene', 'models'))

from disnet import DisNet
from util import config

class OpenSceneModel(object):
	'''
	OpenScene model wrapper.
	
	Arg(s):
	    dataset_name : str
            dataset used
        config_path : str
            path to OpenScene configuration file
        model_restore_path : str
            path or URL to pretrained model weights
        device : torch.device
            device to run model on
	'''

	def __init__(self,
	             dataset_name,
		         config_path,
		         model_restore_path=None,
		         device=torch.device('cuda')):
		         
		self.dataset_name = dataset_name

		self.config = config.load_cfg_from_cfg_file(config_path)
		self.model = DisNet(cfg=self.config)

		if model_restore_path is None and hasattr(self.config, 'model_path'):
			model_restore_path = self.config.model_path

		self.device = device
		self.to(self.device)

		if model_restore_path is not None:
			self.restore_model(model_restore_path=model_restore_path)

		self.eval()

	def forward(self, x):
		'''
		Forwards input through the network

		Arg(s):
			x : MinkowskiEngine.SparseTensor
				sparse 3D input

		Return(s):
			torch.Tensor[float32]
				semantic feature for each input point/voxel
		'''

		return self.model.forward(x)

	def parameters(self):
		'''
		Returns the list of parameters in the model

		Return(s):
			list[torch.Tensor[float32]]
				list of parameters
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
		Loads weights from checkpoint

		Arg(s):
			model_restore_path : str
				path or URL to model weights
			optimizer : torch.optim
				optimizer to restore
		'''

		scheme = urllib.parse.urlparse(model_restore_path).scheme
		is_url = scheme in ('http', 'https')
		
		if is_url:
		    checkpoint = model_zoo.load_url(
		        model_restore_path,
		        progress=True)
		 
		elif os.path.isfile(model_restore_path):
		    checkpoint = torch.load(
		        model_restore_path,
		        map_location=self.device)
		       
		else:
		    raise ValueError('No checkpoint found at: {}'.format(model_restore_path))
		
		self.model.load_state_dict(
		    checkpoint['state_dict'],
		    strict=True)
		

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
