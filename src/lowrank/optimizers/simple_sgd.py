from torch.optim import Optimizer
import torch

class SimpleSGD(Optimizer):
    def __init__(self, parameters, lr=2e-4):
        """
        A simple implementation of Stochastic Gradient Descent (SGD) optimization algorithm.

        This optimizer updates the parameters based on the gradient and a fixed learning rate.
        It's a straightforward implementation suitable for small-scale or simple neural network models.

        Parameters
        ----------
        parameters : iterable
            An iterable of parameters to optimize or dicts defining parameter groups.
        lr : float, optional
            The learning rate to use for optimization. Default is 2e-4.

        Raises
        ------
        ValueError
            If an invalid learning rate (non-positive) is provided.
        """

        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {'lr': lr}
        super().__init__(parameters, defaults)

    def step(self):
        """
        Performs a single optimization step.

        This method iteratively updates each parameter based on its gradient and
        the specified learning rate. The gradients are assumed to be computed 
        prior to calling this method.
        """
        for group in self.param_groups:
            for params in group['params']:
                if params.grad is not None:
                    with torch.no_grad():  # Temporarily set all the requires_grad flags to False
                        delta_params = - params.grad * group['lr']
                        params += delta_params