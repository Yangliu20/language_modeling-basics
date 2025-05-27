import torch.nn as nn
import torch


def cross_entropy_loss(preds: torch.Tensor, targets: torch.Tensor):
    """
    preds dim: (batch_size vocab_size)
    targets dim: (batch_size)
    """
    batch_size = preds.shape[0]
    max_score = torch.max(preds, dim=-1, keepdim=True).values
    preds -= max_score

    loss_per_sample = - preds[torch.arange(batch_size), targets] + torch.log(torch.sum(torch.exp(preds), dim=-1))
    # print(loss_per_sample.shape)
    return torch.mean(loss_per_sample)




if __name__ == "__main__":

    preds = torch.randn((10, 5))
    targets = torch.tensor([2,3,1,0,2,4,1,1,3,2], dtype=torch.long)
    print(preds)
    print(targets)
    print(cross_entropy_loss(preds, targets))