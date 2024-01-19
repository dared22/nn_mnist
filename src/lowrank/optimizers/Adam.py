from torch.optim import Optimizer
import torch

# ---- Todo: Implement Adam ----
class Adam(Optimizer):
	def __init__(self, parameters, lr=3e-4, beta1 = 0.9, beta2 = 0.999):
		"""Initializes the Adam optimizer.

		Args:
			parameters (iterable): Iterable of parameters to optimize.
			lr (float, optional): Learning rate. Default: 3e-4 (Karpathy Constant)
			beta1 (float, optional): Exponential decay rate for the first moment estimates. Default: 0.9
			beta2 (float, optional): Exponential decay rate for the second moment estimates. Default: 0.999
		"""
		if lr <= 0.0:
			raise ValueError(f"Invalid learning rate: {lr}")

		defaults = {'lr': lr, 'beta1': beta1, 'beta2': beta2}
		super().__init__(parameters, defaults)
	
	def step(self, closure=None):
		"""Performs a single optimization step."""
		loss = None
		pass
