import numpy as np
import torch
import random

def data_loading(input_array: np.ndarray, batch_size: int, context_length: int, device: torch.device, seed: int=101):
    """
    A function takes a numpy array (integer array with token IDs), a batch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), 
    and returns a pair of tensors: the sampled input sequences and the corresponding next-token targets.
    """

    # np.random.seed(seed)
    # random.seed(seed)

    input_length = len(input_array)
    sample_indices = np.random.choice(input_length-context_length, batch_size, replace=False)

    in_token_indices = np.array([np.arange(i,i+context_length) for i in sample_indices])
    out_token_indices = np.array([np.arange(i+1,i+1+context_length) for i in sample_indices])

    return torch.tensor(input_array[in_token_indices], device=device), torch.tensor(input_array[out_token_indices], device=device)



if __name__ == "__main__":

    x = np.arange(50)
    in_tensor, out_tensor = data_loading(x, 10, 20, torch.device("cpu"))
    print(in_tensor)
    print(out_tensor)