from torch.optim import Optimizer
import torch

class SimpleSGD(Optimizer):
    def __init__(self, parameters, lr=2e-4):
        """Initializes the SimpleSGD optimizer.

        Args:
            parameters (iterable): Iterable of parameters to optimize.
            lr (float, optional): Learning rate. Default: 2e-4
        """
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {'lr': lr}
        super().__init__(parameters, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for params in group['params']:
                if params.grad is not None:
                    with torch.no_grad():  # Temporarily set all the requires_grad flags to False
                        delta_params = - params.grad * group['lr']
                        params += delta_params
                
        return loss