from collections.abc import Callable, Iterable 
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    """
    SGD optimizer
    """
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure() 
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]: 
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value. 
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place. 
                state["t"] = t + 1 # Increment iteration number.
        return loss



class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer
    """
    def __init__(self, params, lr, betas, weight_decay, eps):
        
        assert lr > 0, ValueError(f"Invalid learning rate: {lr}")
        assert betas[0] > 0 and betas[0] < 1, ValueError(f"Invalid betas[0]: {betas[0]}")
        assert betas[1] > 0 and betas[1] < 1, ValueError(f"Invalid betas[1]: {betas[1]}")
        assert weight_decay > 0, ValueError(f"Invalid weight_decay: {weight_decay}")
        assert eps > 0, ValueError(f"Invalid eps: {eps}")

        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):

        loss = None if closure is None else closure() 

        for group in self.param_groups:

            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]: 
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                first_moment = state.get("first_moment", 0.)
                second_moment = state.get("second_moment", 0.)
                grad = p.grad.data

                # update first moment estimate
                first_moment = beta1 * first_moment + (1 - beta1) * grad
                second_moment = beta2 * second_moment + (1 - beta2) * (grad**2)

                # adjusted learning rate
                lr_adj = lr * math.sqrt(1-beta2**t) / (1-beta1**t)

                # update parameter
                p.data -= lr_adj * first_moment / (torch.sqrt(second_moment) + eps)
                p.data -= lr * weight_decay * p.data

                # update state
                state["t"] = t + 1
                state["first_moment"] = first_moment
                state["second_moment"] = second_moment

        return loss