import os, sys
import torch

sys.path.insert(0, os.path.join('external_src', 'segmentation'))

class TemplateModel(object):
	'''
	Template for interfacing with the models

	Arg(s):
		dataset_name : str
			dataset used
		device : torch.device
			device to run model on
	'''
	def __init__(self,
	             dataset_name,
			     device=torch.device('cuda')):

			self.device = device
			self.to(self.device)

	def forward(self, x):
	    '''
	    Forwards input through network
	    
	    Arg(s):
	        x : input
	    Return(s):
	        torch.Tensor[float32]
	            model output
	    '''
	    
		pass
    
    def parameters(self):
        '''
        Returns the list of parameters in the model
        
        Return(s):
            list[torch.Tensor[float32]] : list of parameters
        '''
        
        pass
        
	def train(self):
	    '''
	    Sets model to training mode
	    '''
	    
		pass

	def eval(self):
	    '''
	    Sets model to evaluation mode
	    '''
	    
		pass

	def to(self, device):
	    '''
	    Move model to a device
	    
	    Arg(s): 
	        device : torch.device
	            device to use
	    '''
	    
		self.device = device
		pass
		
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
        pass
        
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
        
        pass
