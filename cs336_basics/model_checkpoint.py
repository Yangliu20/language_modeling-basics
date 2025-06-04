import torch
import os
import typing
from typing import Union

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]):
    """
    A function that dumps all the states from `model`, `optimizer` and `iteration` into the file-like object `out`
    """
    data = {
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(), 
        "iteration": iteration
    }
    torch.save(data, out)
    print("Checkpoint saved.")
    return

def load_checkpoint(src: Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    A function that loads a checkpoint from `src` and then recovers the `model` and `optimizer` states from that checkpoint.
    """
    data = torch.load(src)
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    print("Model and Optimizer states loaded from checkpoint.")
    return data["iteration"]