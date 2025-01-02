import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math


@dataclass
class GPTConfig:
    n_vocab: int = 50257
    n_block: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    bias: bool = False


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.register_buffer("bias", torch.tril(torch.ones(
            config.n_block, config.n_block)).view(1, 1, config.n_block, config.n_block))

    def forward(self, x):
        # important
        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality(n_embd)
        qkv = self.c_attn(x)
        # split qkv into q, k, v
        q, k, v = qkv.split(self.n_embd, dim=2)
        # fancy q, k, v dimensionality change (why)
        k = k.view(B, T, self.n_head, C //
                   self.n_head). transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C //
                   self.n_head). transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head). transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * \
            (1.0 / math.sqrt(k.size - 1))  # calculate attn
        att = att.masked_fill(
            self.bias[:, :, :T, :T] == 0, float('inf'))  # mask attn
        y = F.softmax(att, dim=-1)  # calculate softmax
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)  # c_attn, c_proj
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)  # c_fn, c_proj

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.n_vocab, config.n_embd),
            wpe=nn.Embedding(config.n_block, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.n_vocab, bias=False)

    @classmethod
    def from_pretrained(cls, model_type):
        # n_vocab, n_block, n_embd, n_head, n_layer
        # we will be fixing n_vocab and n_block
        assert model_type in {'gpt2', 'gpt2-small', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-small': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['n_vocab'] = 50257
        config_args['n_block'] = 1024
        config_args['bias'] = True

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()  # our own model
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(
            '.attn.bias')]  # we are removing bias

        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = model_hf.state_dict()  # model from hugging face
        sd_hf_keys = sd_hf.keys()
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith(
            '.attn.masked_bias')]
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith(
            '.attn.bias')]  # we are removing bias

        # set weights from hugging face to our model
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_hf_keys)
        for k in sd_hf_keys:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd[k].shape == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


print(GPT(GPTConfig))
model = GPT.from_pretrained('gpt2')
print("Didn't crash yay")
# model.to_device()
