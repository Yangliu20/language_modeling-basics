import torch.nn as nn
import torch
import numpy as np
import random
import os
from optimizer import *
from data_loader import *
from utils import *
from model_checkpoint import *
from model import *
from s3_utils import upload_dir
import json
import boto3
import shutil

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
        os.makedirs(self.result_path, exist_ok=True)
        self.ckpt_dir = os.path.join(self.result_path, "model_checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
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
            lr_i = self._get_learning_rate(i+1)
            optimizer_i = AdamW(model.parameters(), lr=lr_i, betas=self.optim_betas, weight_decay=self.optim_weight_decay, eps=self.optim_eps)

            optimizer_i.zero_grad()
            loss_step_i = 0.
            for _ in range(self.gradient_accumulation_steps):
                train_x, train_y = self._get_train_batch()
                train_x = train_x.to(torch.long)
                train_y = train_y.to(torch.long)
                # print(train_x)
                # print(train_y)
                pred_y = model(train_x)
                loss = cross_entropy_loss(preds=pred_y, targets=train_y)
                loss.backward()
                loss_step_i += loss.item()
            optimizer_i.step()
            loss_step_i /= self.gradient_accumulation_steps # average loss per training batch

            if i % self.eval_every_steps == 0 or i == (self.n_steps-1):
                model.eval()
                with torch.no_grad():
                    eval_loss_step_i = 0.
                    factor = 10 if i==(self.n_steps-1) else 1
                    for _ in range(self.n_eval_batches*factor):
                        valid_x, valid_y = self._get_valid_batch()
                        valid_x = valid_x.to(torch.long)
                        valid_y = valid_y.to(torch.long)
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



def parse_config(config_path):

    with open(config_path) as f:
        config_dict = json.load(f)
    model_params = config_dict["model_architecture"]
    training_params = config_dict["model_training"]
    training_params["context_length"] = model_params["context_length"]

    return model_params, training_params

    
def check_params_count(torch_model):
    ### check parameter count
    total_params_cnt = 0
    trainable_params_cnt = 0
    for p in torch_model.parameters():
        total_params_cnt += p.numel()
        if p.requires_grad:
            trainable_params_cnt += p.numel()
    r = trainable_params_cnt/total_params_cnt*100
    print("Count of all parameters:", total_params_cnt)
    print("Count of trainable parameters:", trainable_params_cnt)
    print(f"Percentage of trainable parameters: {r:.2f}%")


def train_model(config_path, train_data_path, valid_data_path, result_path, overwrite, seed):
    
    model_args, training_args = parse_config(config_path)
    print("Training model with:")
    print(model_args)
    print(training_args)

    if os.path.isdir(result_path):
        if not overwrite:
            raise ValueError(f"{result_path} already trained.")
        else:
            shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=True)
    with open(f"{result_path}/model_args.json", 'w') as fp:
        json.dump(model_args, fp)
    with open(f"{result_path}/training_args.json", 'w') as fp:
        json.dump(training_args, fp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device", device)
    model_lm = TransformerLM(
        vocab_size = model_args["vocab_size"], 
        context_length = model_args["context_length"], 
        num_layers = model_args["num_layers"], 
        d_model = model_args["d_model"], 
        num_heads = model_args["num_heads"], 
        d_ff = model_args["d_ff"], 
        rope_theta = model_args["rope_theta"], 
    ).to(device)

    check_params_count(model_lm)

    trainer_lm = trainer(
        model=model_lm, 
        train_data_path=train_data_path, 
        valid_data_path=valid_data_path, 
        training_args=training_args, 
        result_path=result_path, 
        device=device, 
        seed=seed
    )
    trainer_lm.train()

if __name__ == "__main__":

    s3 = boto3.client('s3')

    ## download data from s3
    download_dir = "/home/ec2-user/data_download"
    BUCKET_NAME = "llm-cs336"
    os.makedirs(download_dir, exist_ok=True)
    for dataset in ["train", "valid"]:
        s3_key = f"data/TinyStoriesV2-GPT4-{dataset}-encoded-iterable.npy"
        local_path = f"{download_dir}/TinyStoriesV2-GPT4-{dataset}-encoded-iterable.npy"
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        print(f"{s3_key} downloaded. ")

    train_model(
        config_path = "/home/ec2-user/language_modeling-basics/cs336_basics/model_base_config.json", 
        train_data_path = f"{download_dir}/TinyStoriesV2-GPT4-train-encoded-iterable.npy", 
        valid_data_path = f"{download_dir}/TinyStoriesV2-GPT4-valid-encoded-iterable.npy", 
        result_path = "/home/ec2-user/model/test_run_20250620", 
        overwrite = True, 
        seed=101
    )

    ## upload results to s3
    upload_dir(BUCKET_NAME, directory_path="/home/ec2-user/model/test_run_20250620", s3_prefix="model/test_run_20250620")


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