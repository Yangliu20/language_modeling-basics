import torch.nn as nn
import torch
import json
import numpy as np
import random

from tokenizer import tokenizer
from model import TransformerLM, softmax

class inference_decoder():
    """
    Generate text from a trained LM by inputting a prompt
    """
    def __init__(self, tokenizer: tokenizer, model: nn.Module, 
                 max_num_tokens: int, temperature: float = 1., proba_trunc: float = 1., 
                 end_token: str = "<|endoftext|>", seed: int = 42):
        
        self.tokenizer = tokenizer
        self.model = model
        self.max_num_tokens = max_num_tokens
        self.temperature = temperature
        self.proba_trunc = proba_trunc
        self.end_token = end_token

        self.end_token_id = tokenizer.encode(end_token)[0]

        assert self.temperature > 0, "temperature should be >= 0"
        assert self.proba_trunc > 0 and self.proba_trunc <= 1, "proba_trunc should be in range of (0, 1]"

        ## set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def decode(self, prompt: str):

        encode_ids = self.tokenizer.encode(prompt)
        encode_ids = torch.tensor([encode_ids])

        count_tokens_generated = 0
        while count_tokens_generated < self.max_num_tokens:

            ## get next token scores
            with torch.no_grad():
                out_scores = self.model(encode_ids)[0, -1, :]
            # print(out_scores.shape)
            out_proba = softmax(out_scores/self.temperature, dim=-1)
            sorted_proba, indices = torch.sort(out_proba, descending=True)
            sorted_proba_cumsum = torch.cumsum(sorted_proba, dim=0)

            ## renomalize top-p candidiates
            if self.proba_trunc < 1.:
                trunc_ind = torch.argwhere(sorted_proba_cumsum >= self.proba_trunc).min().item()
                sorted_proba_cumsum = sorted_proba_cumsum[:trunc_ind+1]
                indices = indices[:trunc_ind+1]
                sorted_proba_cumsum /= sorted_proba_cumsum.max()
            # print(sorted_proba_cumsum)

            ## sample from the probability distribution
            random_number = torch.rand(1)
            sample_ind = torch.argwhere(random_number < sorted_proba_cumsum).min().item()
            sample_token = indices[sample_ind]
            # print(sample_token)

            count_tokens_generated += 1            
            encode_ids = torch.cat((encode_ids, torch.tensor([[sample_token]])), dim=-1)
            # print(encode_ids.shape)

            if sample_token == self.end_token_id:
                break
            
        token_list = encode_ids.numpy().tolist()[0]
        output_text = self.tokenizer.decode(token_list[-count_tokens_generated:])
        return output_text
        

def load_model(model_args, model_state_dict): 

    model_lm = TransformerLM(
        vocab_size = model_args["vocab_size"], 
        context_length = model_args["context_length"], 
        num_layers = model_args["num_layers"], 
        d_model = model_args["d_model"], 
        num_heads = model_args["num_heads"], 
        d_ff = model_args["d_ff"], 
        rope_theta = model_args["rope_theta"], 
    )#.to(device)
    model_lm.load_state_dict(model_state_dict)
    model_lm.eval()
    print("Model loaded.")
    return model_lm


if __name__ == "__main__":

    ## tokenizer setup
    bpe_train_path = "/home/ec2-user/bpe-tokenizer/"
    tokenizer_bpe = tokenizer.from_files(
        vocab_filepath=f'{bpe_train_path}vocab.json', 
        merges_filepath=f'{bpe_train_path}merges.txt', 
        special_tokens=["<|endoftext|>"]
    )

    ## model setup
    run_version = "test_run_20250622"
    local_result_path = f"/home/ec2-user/model/{run_version}"
    with open(f"{local_result_path}/model_args.json") as fp:
        model_args = json.load(fp)
    model_state_dict = torch.load(f"{local_result_path}/model_weights.ckpt")
    model_lm = load_model(model_args, model_state_dict)

    ## decoder
    max_num_tokens = 10
    temperature = 0.5
    proba_trunc = 0.6

    lm_decoder = inference_decoder(
        tokenizer=tokenizer_bpe, 
        model=model_lm, 
        max_num_tokens=max_num_tokens, 
        temperature=temperature, 
        proba_trunc=proba_trunc, 
    )

    prompt = "Give me a short story."
    output = lm_decoder.decode(prompt)
    print("Prompt:", prompt)
    print("Output:", output)