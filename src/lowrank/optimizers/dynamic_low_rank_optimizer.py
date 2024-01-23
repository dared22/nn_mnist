from torch.optim import Optimizer
import torch

class DynamicLowRankOptimizer(Optimizer):
    """
    A specialized optimizer for training DynamicLowRankLayer in neural networks.

    This optimizer is designed to update the parameters of a DynamicLowRankLayer,
    which includes matrices U, S, V, and a bias term. It supports the option to
    toggle training between updating all parameters and updating only the S matrix.

    Parameters
    ----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
    lr : float, optional
        The learning rate to use for optimization. Default: 2e-4.
    toggle_S_training : bool, optional
        If True, toggles the training to focus only on the S matrix in alternating steps.
        Default: False.

    Raises
    ------
    ValueError
        If an invalid learning rate is provided.
    """
    def __init__(self, params, lr=2e-4, toggle_S_training = False):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {'lr': lr, "toggle_S_training": toggle_S_training, "only_S": False}
        super().__init__(params, defaults)

    def step(self, only_S = None):
        """
        Performs a single optimization step.

        This method updates the U, S, V matrices, and the bias term of the
        DynamicLowRankLayer, with an option to update only the S matrix.

        Parameters
        ----------
        only_S : bool, optional
            If specified, overrides the internal 'only_S' setting for this step,
            determining whether only the S matrix should be updated. If None, uses
            the internal setting. Default: None.

        Raises
        ------
        ValueError
            If any of the parameters (U, S, V, bias) are not found in the optimizer parameters.

        Notes
        -----
        - The update rules for U, S, V, and bias are specific to the structure of
          the DynamicLowRankLayer.
        - If 'toggle_S_training' is True, the optimizer alternates between updating
          only S and updating all parameters in subsequent steps.
        """
        U, S, V, bias = None, None, None, None  # Initialize variables
        for group in self.param_groups:
            U = group['params'][0]
            S = group['params'][1]
            V = group['params'][2]
            bias = group['params'][3]
        
        # If only_S is not specified, use value stored by the internal switching mechanism. This makes it possible for the optimizer to be used standalone or as part of a MetaOptimizer.
        if only_S is None:
            only_S = self.defaults["only_S"]
        
        # If only_S is specified, overwrite/update the internal switching mechanism
        else:
            self.defaults["only_S"] = only_S

        # Check if all variables are assigned
        if U is None or S is None or V is None or bias is None:
            raise ValueError("U, S, V, or bias not found in optimizer parameters.")

        # Update U, S, V using the existing gradients
        if only_S:
            with torch.no_grad():
                S_new = S - self.defaults["lr"] * S.grad
                S.data = S_new

        else:
            with torch.no_grad():   

                # 1. Update U
                K_old = U @ S
                grad_U = U.grad
                K_new = K_old - self.defaults["lr"] * grad_U @ S
                U_new, R = torch.linalg.qr(K_new, 'reduced')
                M = U_new.t() @ U

                U.data = U_new

                # 2. Update V
                L_old = V @ S.t()
                L_new = L_old - self.defaults["lr"] * V.grad @ S.t()
                V_new, _ = torch.linalg.qr(L_new, 'reduced')
                N = V_new.t() @ V # Doesn't work if I transpose V_new here? 
                V.data = V_new # Loss becomes nan if I don't use .data

                # 3. Update S temporarily
                S_tilde = M @ S @ N.t() # Doesn't work if I transpose N here?
                S_new = S_tilde - self.defaults["lr"] * S.grad
                S.data = S_new

                # 4. Update bias using normal SGD
                bias.data = bias - self.defaults["lr"] * bias.grad
                
        if self.defaults["toggle_S_training"]:
            self.toggle_only_S()

    def toggle_only_S(self):
        """
        Toggles the internal 'only_S' setting.

        When toggled, the optimizer will switch between updating only the S matrix
        and updating all parameters in subsequent steps.
        """
        self.defaults["only_S"] = not self.defaults["only_S"]