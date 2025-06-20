import torch.nn as nn
import torch
import numpy as np
import random
import os
from cs336_basics.optimizer import *
from cs336_basics.data_loader import *
from cs336_basics.utils import *
from cs336_basics.model_checkpoint import *

class trainer():
    """
    training pipeline
    """
    def __init__(self, model: nn.Module, train_data_path: str, valid_data_path: str, training_args: dict, 
                 result_path: str, device: torch.device, seed: int = 101):

        self.model = model.to(device)
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.training_args = training_args
        self.result_path = result_path
        self.seed = seed
        self.device = device

        ## set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self._parse_training_args()
        self._create_local_result_dir()

    
    def _parse_training_args(self):
        """
        function to parse training args dict.
        A sample training_args dict: 
        {
            "n_steps": 100, 
            "batch_size": 2048,
            "gradient_accumulation_steps": 4, 
            "n_eval_batches": 4, 
            "eval_every_steps": 5, 
            "context_length": 500, 
            "optimizer": {
                "betas": [0.9, 0.999],
                "weight_decay": 0.1,
                "eps": 1e-8, 
            }, 
            "lr_scheduler": {
                "lr_max": 5e-4,
                "lr_min": 1e-5,
                "warmup_iters": 10,
                "cosine_cycle_iters": 50,
            }, 
            "grad_clip": {
                "max_l2_norm": 1.0,
            }
        }
        """
        # optimizer
        # self.optim_lr = self.training_args["optimizer"]["lr"]
        self.optim_betas = self.training_args["optimizer"]["betas"]
        self.optim_weight_decay = self.training_args["optimizer"]["weight_decay"]
        self.optim_eps = self.training_args["optimizer"]["eps"]
        
        # lr scheduler
        self.lr_scheduler_max = self.training_args["lr_scheduler"]["lr_max"]
        self.lr_scheduler_min = self.training_args["lr_scheduler"]["lr_min"]
        self.lr_scheduler_warmup_iters = self.training_args["lr_scheduler"]["warmup_iters"]
        self.lr_scheduler_cosine_cycle_iters = self.training_args["lr_scheduler"]["cosine_cycle_iters"]

        # gradient clipping
        self.grad_clip_max_l2_norm = self.training_args["grad_clip"]["max_l2_norm"]

        self.n_steps = self.training_args["n_steps"]
        self.batch_size = self.training_args["batch_size"]
        self.gradient_accumulation_steps = self.training_args["gradient_accumulation_steps"]
        self.n_eval_batches = self.training_args["n_eval_batches"]
        self.eval_every_steps = self.training_args["eval_every_steps"]
        self.context_length = self.training_args["context_length"]

    def _create_local_result_dir(self):

        # if os.is_dir(self.result_path):
        #     raise ValueError(f"{self.result_path} already trained.")
        os.makedirs(self.result_path)
        self.ckpt_dir = os.path.join(self.result_path, "model_checkpoints")
        os.makedirs(self.ckpt_dir)
        self.log_txt_path = os.path.join(self.result_path, "training_log.txt")

        print(self.ckpt_dir)
        print(self.log_txt_path)
    
    def _get_learning_rate(self, curr_iter):

        return learning_rate_cosine_schedule(curr_iter, 
                                             lr_max=self.lr_scheduler_max, 
                                             lr_min=self.lr_scheduler_min, 
                                             warmup_iters=self.lr_scheduler_warmup_iters, 
                                             cosine_cycle_iters=self.lr_scheduler_cosine_cycle_iters)
    
    def _get_train_batch(self):

        return data_loading(self.train_data, batch_size=self.batch_size, context_length=self.context_length, device=self.device)
    
    def _get_valid_batch(self):

        return data_loading(self.valid_data, batch_size=self.batch_size, context_length=self.context_length, device=self.device)
    
    def _log_metrics(self, log_dict):

        log_str = f"\nIteration {log_dict['iter']}, train loss {log_dict['train_loss']}"
        if "valid_loss" in log_dict:
            log_str += f", valid loss {log_dict['valid_loss']}"
        log_str += f", learning rate {log_dict['lr']}"
        print(log_str)
        with open(self.log_txt_path, "w") as file:
            file.write(log_str+"\n")

    def train(self):
        
        model = self.model
        self.train_data = np.load(self.train_data_path, mmap_mode='r')
        self.valid_data = np.load(self.valid_data_path, mmap_mode='r')

        for i in range(self.n_steps):

            model.train()
            lr_i = self._get_learning_rate(i)
            optimizer_i = AdamW(model.parameters(), lr=lr_i, betas=self.optim_betas, weight_decay=self.optim_weight_decay, eps=self.optim_eps)

            optimizer_i.zero_grad()
            loss_step_i = 0.
            for _ in range(self.gradient_accumulation_steps):
                train_x, train_y = self._get_train_batch()
                pred_y = model(train_x)
                loss = cross_entropy_loss(preds=pred_y, targets=train_y)
                loss.backward()
                loss_step_i += loss.item()
            optimizer_i.step()
            loss_step_i /= self.gradient_accumulation_steps # average loss per training batch

            if i % self.eval_every_steps == 0:
                model.eval()
                with torch.no_grad():
                    eval_loss_step_i = 0.
                    for _ in range(self.n_eval_batches):
                        valid_x, valid_y = self._get_valid_batch()
                        valid_pred_y = model(valid_x)
                        vloss = cross_entropy_loss(preds=valid_pred_y, targets=valid_y)
                        eval_loss_step_i += vloss
                    eval_loss_step_i /= self.n_eval_batches # average loss per validation batch
            
            log_dict = {
                "iter": i, 
                "lr": lr_i, 
                "train_loss": loss_step_i, 
                **({"valid_loss": eval_loss_step_i} if i % self.eval_every_steps == 0 else {})
            }
            self._log_metrics(log_dict)
            save_checkpoint(model=model, optimizer=optimizer_i, iteration=i, out=os.path.join(self.ckpt_dir, f"checkpoint_{i}.ckpt"))

        print("Training finished.")
        model_path = os.path.join(self.result_path, f"model_weights.ckpt")
        torch.save(model.state_dict(), model_path)






# model:
# vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float

# optimizer:
# lr, betas, weight_decay, eps

# lr scheduler:
# lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int

# gradient clipping:
# max_l2_norm

# n_steps
# gradient_accumulation_steps
# n_eval_batches
# eval_every_steps
# batch_size