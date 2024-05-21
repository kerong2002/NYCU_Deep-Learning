import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, _ = self.vqgan.encode(x)
        return codebook_mapping, codebook_indices.reshape(codebook_mapping.shape[0], -1)
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "sqrt":
            return lambda r: 1 - np.sqrt(r)
        elif mode == "constant":
            return lambda r: 1
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        # x: (b, c, h, w)
        _, z_indices = self.encode_to_z(x)
        
        mask_token = torch.ones(z_indices.shape, device=z_indices.device).long() * self.mask_token_id 
        mask = torch.bernoulli(0.5 * torch.ones(z_indices.shape, device=z_indices.device)).bool()
        
        new_indices = mask * mask_token + (~mask) * z_indices
        logits = self.transformer(new_indices)
        z_indices=z_indices # ground truth
        logits = logits  # transformer predict the probability of tokens
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding
    @torch.no_grad()
    def inpainting(self, z_indices, mask, mask_num, ratio, mask_func):
        z_indices_with_mask = mask * self.mask_token_id + (~mask) * z_indices
        logits = self.transformer(z_indices_with_mask)
        probs = torch.softmax(logits, dim=-1)
        
        # make sure the predict token is not mask token
        z_indices_predict = torch.distributions.categorical.Categorical(logits=logits).sample()
        while torch.any(z_indices_predict == self.mask_token_id):
            z_indices_predict = torch.distributions.categorical.Categorical(logits=logits).sample()
            
        z_indices_predict = mask * z_indices_predict + (~mask) * z_indices

        # get prob from predict z_indices
        z_indices_predict_prob = probs.gather(-1, z_indices_predict.unsqueeze(-1)).squeeze(-1)
        z_indices_predict_prob = torch.where(mask, z_indices_predict_prob, torch.zeros_like(z_indices_predict_prob) + torch.inf)

        mask_ratio = self.gamma_func(mask_func)(ratio)
        
        mask_len = torch.floor(mask_num * mask_ratio).long()

        g = torch.distributions.gumbel.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices_predict_prob.device)
        temperature = self.choice_temperature * (1 - mask_ratio)
        confidence = z_indices_predict_prob + temperature * g
        sorted_confidence = torch.sort(confidence, dim=-1)[0]
        cut_off = sorted_confidence[:, mask_len].unsqueeze(-1)
        new_mask = (confidence < cut_off)
        return z_indices_predict, new_mask
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
