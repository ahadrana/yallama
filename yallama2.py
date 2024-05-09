from pathlib import Path
import os,sys,math
import pprint
from dataclasses import dataclass
from typing import (Optional,Generator)
import contextlib
import torch
from torch import nn
from sentencepiece import SentencePieceProcessor
from safetensors.torch import load_model, safe_open
from tqdm import tqdm
import torch.nn.functional as F
from debug import save_tensor
import matplotlib.pyplot as plt
import numpy as np



MAX_CONTEXT = int(os.getenv("MAX_CONTEXT", "4096"))


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"    
print(f"Using {get_device()} device")


"""
Layers Layers Layers

"""


# From Tinygrad!!!
# calculating params:
# traditionally, the MLP in the transformer architecture has hidden_dim = dim*4 [arxiv/1706.03762, 3.3]
# however, Llama uses SwiGLU. in order to preserve param count to original transformer arch, hidden_dim must be = 2/3 * (dim*4) [arxiv/2002.05202]
# for models using MQA (n_kv_heads != n_heads), preserving param count means hidden dim must be further multiplied by 1.3 [arxiv/2307.09288, A.2.1]
MODEL_PARAMS = {
  "1": {
    "7B": {
      "args": {"dim": 4096, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": 32000, "hidden_dim": 11008},
      "files": 1,
    },
    "13B": {
      "args": {"dim": 5120, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-06, "vocab_size": 32000, "hidden_dim": 13824},
      "files": 2,
    },
    "30B": {
      "args": {"dim": 6656, "n_heads": 52, "n_layers": 60, "norm_eps": 1e-06, "vocab_size": 32000, "hidden_dim": 17920},
      "files": 4,
    },
    "65B": {
      "args": {"dim": 8192, "n_heads": 64, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 22016},
      "files": 8,
    },
  },
  "2": {
    "7B": {
      "args": {"dim": 4096, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 11008},
      "files": 1,
    },
    "13B": {
      "args": {"dim": 5120, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 13824},
      "files": 2,
    },
    "70B": {
      "args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 28672},
      "files": 8,
    },
  },
  "code": {
    "7B": {
      "args": {"dim": 4096, "n_layers": 32, "n_heads": 32, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32016, "hidden_dim": 11008},
      "files": 1,
    },
    "7B-Python": {
      "args": {"dim": 4096, "n_layers": 32, "n_heads": 32, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32000, "hidden_dim": 11008},
      "files": 1,
    },
    "7B-Instruct": {
      "args": {"dim": 4096, "n_layers": 32, "n_heads": 32, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32016, "hidden_dim": 11008},
      "files": 1,
    },
    "13B": {
      "args": {"dim": 5120, "n_layers": 40, "n_heads": 40, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32016, "hidden_dim": 13824},
      "files": 2,
    },
    "13B-Python": {
      "args": {"dim": 5120, "n_layers": 40, "n_heads": 40, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32000, "hidden_dim": 13824},
      "files": 2,
    },
    "13B-Instruct": {
      "args": {"dim": 5120, "n_layers": 40, "n_heads": 40, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32016, "hidden_dim": 13824},
      "files": 2,
    },
    "34B": {
      "args": {"dim": 8192, "n_layers": 48, "n_heads": 64, "n_kv_heads": 8, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32000, "hidden_dim": 22016},
      "files": 4,
    },
    "34B-Python": {
      "args": {"dim": 8192, "n_layers": 48, "n_heads": 64, "n_kv_heads": 8, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32000, "hidden_dim": 22016},
      "files": 4,
    },
    "34B-Instruct": {
      "args": {"dim": 8192, "n_layers": 48, "n_heads": 64, "n_kv_heads": 8, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32000, "hidden_dim": 22016},
      "files": 4,
    },
  },
  "tiny": {
    "1B": {
      "args": {"dim": 2048, "n_layers": 22, "n_heads": 32, "n_kv_heads": 4, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 5632},
      "files": 1,
    },
    "1B-Chat": {
      "args": {"dim": 2048, "n_layers": 22, "n_heads": 32, "n_kv_heads": 4, "norm_eps": 1e-05, "vocab_size": 32003, "hidden_dim": 5632},
      "files": 1,
    }
  }
}

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads:int
    vocab_size: int
    hidden_dim: int
    norm_eps: float = 1e-5
    rope_theta: int = 10_000
    max_context_size : int = 2048
    # Needed for KV cache
    max_batch_size: int = 32
    # llama2 tokenizer doesn't specify the pad id properly
    pad_id: int = 0
    
    device: str = None
    flash: bool = False
    
    def __init__(self,model_params,training,max_context_size,max_batch_size,device="cuda"): 
        self.training = training
        self.device = device  # Set device first, might be overridden by model_params
        self.max_context_size = max_context_size
        self.max_batch_size = max_batch_size
        for key, value in model_params['args'].items():
            setattr(self, key, value)
        # use scaled dot product attention if available
        self.use_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')



def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)

def _permute(t,dim,head_dim,n_heads):
    return (
        t.view(n_heads, 2, head_dim // 2, dim)
        .transpose(1, 2)
        .reshape((head_dim * n_heads), dim)
    )

class LLaMA(nn.Module):
  
    @staticmethod
    def build(model_path,tokenizer_path,model_gen="2", model_size="7B",training=True,max_context=MAX_CONTEXT,max_batch_size=32,device=None):
        device = get_device() if device is None else device
        args = ModelArgs(MODEL_PARAMS[model_gen][model_size],training,max_context,max_batch_size,device)
        tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
        #assert tokenizer.vocab_size() == args.vocab_size, f"{tokenizer.vocab_size()=} not equal to {args.vocab_size}"
        # TODO: TYPE SHOULD BE CONFIGURABLE
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        model = LLaMA(args,tokenizer).to(device, dtype=torch.bfloat16)
        # torch.set_default_dtype(old_dtype)
        
        if model_path.is_dir():
          safetensors_path = f"{model_path}/model.safetensors"
          if os.path.exists(safetensors_path):
              pp = pprint.PrettyPrinter(indent=4)
              #print("Model state_dict:")
              #pp.pprint(model.state_dict())
              loaded_state = {}
              with safe_open(safetensors_path, "pt", device="cpu") as f:
                for k in f.keys():
                    loaded_state[k] = f.get_tensor(k)
                    # NOTE: HF stores the weights in a different format due to a different 
                    # ROPE implementation !!!
                    if "q_proj" in k:
                      loaded_state[k] = _permute(loaded_state[k],args.dim,args.dim // args.n_heads,args.n_heads)
                    elif "k_proj" in k:
                      loaded_state[k] = _permute(loaded_state[k],args.dim,args.dim // args.n_heads,args.n_kv_heads)
                      
              state_dict = loaded_state
              
              k_proj_0_weights = state_dict["model.layers.0.self_attn.k_proj.weight"]
              save_tensor(k_proj_0_weights, 0, "k_proj_0_weights")
              
              #print("Returned dict from load:")
              #pp.pprint(state_dict)
              model.load_state_dict(state_dict)
          else:
              raise FileNotFoundError(f"model.safetensors not found in {model_path}")
        return model
      
    def __init__(self,model_args:ModelArgs,tokenizer):
        super().__init__()
        
        self.args = model_args
        self.model = Transformer(model_args,model_args.max_context_size)
        self.tokenizer : SentencePieceProcessor = tokenizer
        self.lm_head = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.pad_id = model_args.pad_id
        
    def forward(self,tokens: torch.Tensor, start_pos: int):
      # (B, Seq_Len) -> (B, Seq_Len, Dim)
      outputs = self.model(tokens, start_pos)
      # (B, Seq_Len, Dim) -> (B, Seq_Len, Vocab_Size)
      logits = self.lm_head(outputs).float()
  
      return logits
      
    def generate(self, prompts: list[str], temperature: float = 0.6, top_k = 300, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_context_size - 1

        device = self.args.device
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Check batch size limits 
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_context_size, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_context_size, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        tokens = torch.full((batch_size, total_len), self.pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        
        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != self.pad_id # True if the token is a prompt token, False otherwise
        
        print("Generating First Token")
        # generate the first token by conditioning on the input prompt
        next_token = self.generate_next_token(
            # from 0
            0,
            # tokens is already BxS
            tokens[:,:max_prompt_len],
            temperature=temperature,
            top_k=top_k,
        ).clone()
        cur_pos = max_prompt_len
        tokens[:, cur_pos] = next_token
        
        cur_iterator = tqdm(range(cur_pos, total_len), desc="Generating subsequent tokens")
        
        for cur_pos in cur_iterator:
            next_token = self.generate_next_token(cur_pos,tokens[:,cur_pos:cur_pos+1],temperature=temperature,top_k=top_k)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            print(f"Prompt {prompt_index}: {self.tokenizer.decode(current_prompt_tokens)}")
            
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)
    
    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p 
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0 
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token

    @staticmethod
    def multinomial_sample_one(probs):
      q = torch.empty_like(probs).exponential_(1)
      return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

    @staticmethod
    def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        # scale the logits based on temperature
        logits = logits / max(temperature, 1e-5)

        # keep only the top_k logits if this is specified
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # select the very last value from the top_k above as the pivot
            pivot = v.select(-1, -1).unsqueeze(-1)
            # set everything smaller than pivot value to inf since these
            # should be pruned
            logits = torch.where(logits < pivot, -float("Inf"), logits)

        # compute the probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # sample the next token
        token = LLaMA.multinomial_sample_one(probs)
        return token


    def generate_next_token(
        self,
        start_pos: int,
        x: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        # x: [B, S]
        # input_pos: [S]
        with torch.no_grad():
          logits = self(x, start_pos)
        
        logits = logits.cpu()

        # logits: [B, s, v] where v is vocab_size
        # for sampling we extract the logits for the
        # last token and convert to shape: [v]
        logits = logits[0, -1]
        # get the last index of the last dimension in the x tensor
        last_index = x.shape[-1] - 1
        save_tensor(logits, 0, f"logits_{last_index}")
        
        # sample the next token
        token = LLaMA.sample(logits, temperature, top_k)
        return token



@staticmethod
# precompute positional frequencies for rotary embeddings
# as defined in paragraph 3.2.2 of the paper - https://arxiv.org/pdf/2104.09864.pdf
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int,theta: float, device: str) -> torch.Tensor:
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device) # (Dim / 2)

    # Construct the positions (the "m" parameter) up to seq_len
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

@staticmethod
def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x)


class Transformer(nn.Module):
  def __init__(self, model_args: ModelArgs, max_context: int = MAX_CONTEXT):
    super().__init__()
    
    self.model_args = model_args
    self.embed_tokens = nn.Embedding(model_args.vocab_size, model_args.dim)
    
    self.layers = nn.ModuleList()
    for i in range(model_args.n_layers):
      self.layers.append(TransformerBlock(model_args, max_context))
      
    self.norm = RMSNorm(model_args.dim, model_args.norm_eps)
    
    self.max_context = max_context
    
    self.freqs_cis = precompute_theta_pos_frequencies(model_args.dim // model_args.n_heads, self.max_context * 2, model_args.rope_theta,model_args.device)
    
    if not model_args.use_sdpa:
      # do it the karpathy way
      print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
      # create a mask for causal attention
      self.register_buffer("bias",torch.tril(torch.ones(max_context, max_context, dtype=torch.bool))
                          .view(1, 1, max_context, max_context))        

  def forward(self, tokens: torch.Tensor, start_pos: int):
    # if not using sdpa and training
    # we want
    # we want
    
    # (B, Seq_Len)
    batch_size, seq_len = tokens.shape

    # (B, Seq_Len) -> (B, Seq_Len, Dim)
    h = self.embed_tokens(tokens)

    
    # establish mask for non-sdpa attention 
    mask = self.bias if not self.model_args.use_sdpa else None
      
    # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
    assert start_pos + seq_len <= self.max_context * 2, f"start_pos + seq_len must be less than or equal to {self.max_context * 2}"
    freqs_complex = self.freqs_cis[start_pos:start_pos + seq_len]
    
    # Consecutively apply all the encoder layers
    for layer in self.layers:
        h = layer(h,start_pos, freqs_complex, mask)
    h = self.norm(h)
    
    return h
 
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter is learned
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        x_fp32 = x.float()
        x_normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_normed.type_as(x)

    def forward(self, x: torch.Tensor):
        # Broadcast Dim across the last dimension
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        x_= self.weight * self._norm(x)
        # retain source type on exit        
        return x_

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs,max_context:int):
        super().__init__()
        
        self.self_attn = SelfAttention(args, max_context)
        self.mlp = FeedForward(args) 
        self.input_layernorm = RMSNorm(args.dim, args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, args.norm_eps)
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape  # (B, Seq_Len, Dim)

        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        x_norm = self.input_layernorm(x)
        h = x + self.self_attn.forward(x_norm, start_pos, freqs_complex,mask)
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.mlp.forward(self.post_attention_layernorm(h))
        return out


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs, max_context: int):
        super().__init__()

        self.model_args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.max_context = max_context
        
        self.q_proj = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_context_size, self.n_kv_heads, self.head_dim)).to(device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_context_size, self.n_kv_heads, self.head_dim)).to(args.device)

    def forward(self,x: torch.Tensor,start_pos: int,freqs_complex: torch.Tensor, mask: Optional[torch.Tensor] = None):
      
        # we rely on kv cache and hence seq_len should normally be 1
        batch_size, seq_len, _ = x.shape  # (B, Seq_Len, Dim)

        # (B, Seq_Len , Dim) -> (B, Seq_Len, H_Q * Head_Dim)
        xq = self.q_proj(x)
        # (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)
        xk = self.k_proj(x)
        # (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)
        xv = self.v_proj(x)

        # (B, Seq_Len, H_Q * Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (B, Seq_Len, H_Q, Head_Dim) --> (B, Seq_Len, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (B, Seq_Len, H_KV, Head_Dim) --> (B, Seq_Len, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # get the seq_len sized tensor from the cache
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Since every group of Q shares the same K and V heads, 
        # just repeat the K and V heads for every Q in the same group.
        
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)
        
        # (B, Seq_Len, H_Q, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)
        
        
        if self.model_args.use_sdpa:
          # use efficient self attention (flash-attention 2 on CUDA at least)
          # TODO: add dropout
           output = torch.nn.functional.scaled_dot_product_attention(xq, keys, values, attn_mask=None, dropout_p=0, is_causal=True)
        else:
          # (B, H_Q, Seq_Len, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q,  Seq_Len, Seq_Len_KV)
          scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
          # convert to float for numerical stability of softmax
          scores = scores.float()
          # apply mask 
          if mask is not None:
            # (B, H_Q,  Seq_Len, Seq_Len_KV) -> (B, H_Q,  Seq_Len, Seq_Len_KV)
            scores = scores.masked_fill(mask[::seq_len,seq_len] == 0, float("-inf"))                      
          # (B, H_Q,  Seq_Len, Seq_Len_KV) -> (B, H_Q,  Seq_Len, Seq_Len_KV)
          scores = F.softmax(scores, dim=-1).type_as(xq)
          # (B, H_Q, Seq_Len, Seq_Len_KV) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
          output = torch.matmul(scores, values)
        
        # (B, H_Q, Seq_Len, Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim) -> (B, Seq_Len, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        # (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        output = self.o_proj(output) 
        
        output_debug = output.flatten()
        save_tensor(output_debug, 0, f"attn_output_{0}_{seq_len - 1}")
        
        return output

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
      
        # source tensor 
        # (B, Seq_Len, N_KV_Heads, Head_Dim)
        
        # add a new dimension to accomodate the n_reps
        
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        
        # expand the tensor to have n_reps 
        # (duplicating the tensor n_reps times along the new dimension)
        # without copying the data
        
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        
        # reshape the tensor to have the desired shape
        
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )
    
class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()
        # From Tinygrad
        # traditionally, the MLP in the transformer architecture has hidden_dim = dim*4 [arxiv/1706.03762, 3.3]
        # however, Llama uses SwiGLU. in order to preserve param count to original transformer arch, hidden_dim must be = 2/3 * (dim*4) [arxiv/2002.05202]
        # for models using MQA (n_kv_heads != n_heads), preserving param count means hidden dim must be further multiplied by 1.3 [arxiv/2307.09288, A.2.1]

        # precalculated for now 
        hidden_dim = args.hidden_dim

        self.gate_proj = nn.Linear(args.dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, args.dim, bias=False)
        self.up_proj = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.gate_proj(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.up_proj(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.down_proj(x)
        
        return x

 
