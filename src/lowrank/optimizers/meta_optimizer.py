from torch.optim import Optimizer
import torch
import torch.nn as nn
from lowrank.layers.dynamic_low_rank import DynamicLowRankLayer
from lowrank.optimizers.dynamic_low_rank_optimizer import DynamicLowRankOptimizer
from lowrank.optimizers.simple_sgd import SimpleSGD

class MetaOptimizer():
    """
    A meta-optimizer for handling different optimization strategies for various layers of a neural network model.

    This optimizer allows for different optimization algorithms and parameters to be used for different types of layers.
    It supports both standard and alternating training modes.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be optimized.
    optimizer_config : dict, optional
        A dictionary mapping layer types to their corresponding optimizer classes and arguments.
        Defaults to using SimpleSGD for all layers if not provided.
    alternating_training : bool, optional
        Indicates whether to use alternating training, where only certain parameters are updated in each step.
        Defaults to True.

    Attributes
    ----------
    model : torch.nn.Module
        The neural network model being optimized.
    layer_optimizers : dict
        A dictionary mapping layer names to their corresponding optimizer instances.
    default_optimizer_params : list
        List of parameters for layers that are not explicitly specified in the optimizer configuration.
    default_optimizer : Optimizer
        The default optimizer used for layers not specified in the optimizer configuration.
    alternating_training : bool
        Whether the optimizer is in alternating training mode.
    train_only_S : bool
        In alternating mode, indicates if only the 'S' parameters should be updated in the current step.
    """
    def __init__(self, model, optimizer_config = None, alternating_training = True):
        """
        Initializes the MetaOptimizer with the specified model, optimizer configuration, and training mode.

        This method sets up individual optimizers for each layer in the model according to the provided
        optimizer configuration. It also prepares for alternating training if specified.

        Note: The optimizer configuration should be a dictionary mapping layer types to optimizer classes
        and their arguments.
        """
        self.model = model
        self.layer_optimizers = {}
        self.default_optimizer_params = []
        self.default_optimizer = None
        self.alternating_training = alternating_training
        if alternating_training:
            self.train_only_S = False

        if optimizer_config is None:
            optimizer_config = {
                "default": (SimpleSGD, {'lr': 3e-4}),
                DynamicLowRankLayer: (DynamicLowRankOptimizer, {'lr': 3e-4}),
                # Other layer types and their optimizers (if any)
            }
        
        for name, layer in model.named_modules():
            if layer == model: # Skip the model itself so it doesn't get assigned to default optimizer (which breaks everything)
                continue

            layer_type = type(layer)
            if layer_type in optimizer_config:
                optimizer_class, optimizer_args = optimizer_config[layer_type]

                # Check if the layer is of type DynamicLowRankLayer
                if isinstance(layer, DynamicLowRankLayer):
                    # Directly pass the parameters of the DynamicLowRankLayer for safety (optimizer doesn't know which parameters are which due to scope stuff, so if the parameters aren't passed in the right order, the optimizer *might* unpack them incorrectly and subsequently break)
                    layer_params = [layer.U, layer.S, layer.V, layer.bias]
                    self.layer_optimizers[name] = optimizer_class(layer_params, **optimizer_args)
                else:
                    self.layer_optimizers[name] = optimizer_class(layer.parameters(), **optimizer_args)

            else:
                # Check if the layer has parameters
                if any(layer.named_parameters()):
                    # print(layer)
                    self.default_optimizer_params.extend(layer.parameters())
            if len(self.default_optimizer_params) > 0 and self.default_optimizer is None:
                optimizer_class, optimizer_args = optimizer_config["default"]
                self.default_optimizer = optimizer_class(layer.parameters(), **optimizer_args)


    def step(self):
        """
        Perform an optimization step for each layer in the model.

        This method updates the parameters of each layer using its respective optimizer.
        In alternating training mode, it alternates between updating only certain parameters
        and updating all parameters.
        """
        if self.alternating_training:
            self.alternating_step()
        else:
            self.standard_step()
                
    def alternating_step(self):
        """
        Perform an optimization step in alternating training mode.

        This method updates either only the 'S' parameters or all parameters for each layer,
        depending on the current state of alternating training.
        """
        if self.train_only_S:
            for key, optimizer in self.layer_optimizers.items():
                if isinstance(optimizer, DynamicLowRankOptimizer):
                    optimizer.defaults["only_S"] = True # Ensure that only S is updated
                    optimizer.step()

            self.toggle_only_S()

        else: 
            for key, optimizer in self.layer_optimizers.items():
                if isinstance(optimizer, DynamicLowRankOptimizer):
                    optimizer.defaults["only_S"] = False # Ensure that everything is updated
                optimizer.step()
        
            if self.default_optimizer:
                self.default_optimizer.step()
            
            self.toggle_only_S()
    
    def standard_step(self):
        """
        Update all parameters in the model.
        """
        for key, optimizer in self.layer_optimizers.items():
            optimizer.step()
        
        if self.default_optimizer:
            self.default_optimizer.step()
        

    def toggle_only_S(self):
        """
        Toggle the state of training only 'S' parameters in alternating training mode.
        """
        self.train_only_S = not self.train_only_S
        

    def zero_grad(self):
        """
        Clear all gradients in the model.
        """
        self.model.zero_grad()