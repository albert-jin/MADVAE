"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    # embd_pdrop = 0.1
    # resid_pdrop = 0.1
    # attn_pdrop = 0.1
    embd_pdrop = 0. 
    resid_pdrop = 0. 
    attn_pdrop = 0.  
    
    conv_kernel_size = 3
    conv_groups = 1
    window_size = 3 

     # eVAE相关参数
    population_size = 20
    elite_ratio = 0.3 
    mutation_rate = 0.05
    crossover_rate = 0.03
    generations = 3
    def __init__(self, state_size, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.state_size = state_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class VAE(nn.Module):
    

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.latent_dim = config.n_embd // 2  
        self.encoder = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd),
            nn.ReLU(),
            nn.Linear(self.n_embd, self.n_embd)
        )
        self.fc_mu = nn.Linear(self.n_embd, self.latent_dim)
        self.fc_var = nn.Linear(self.n_embd, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.n_embd),
            nn.ReLU(),
            nn.Linear(self.n_embd, self.n_embd)
        )
        self.resid_drop = nn.Dropout(config.resid_pdrop)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        reconstructed = reconstructed.view(B, T, C)
        output = self.resid_drop(reconstructed)
        return output


class VAE2(nn.Module):
    

    def __init__(self, config):
        super(VAE2, self).__init__()

        self.config = config
        self.input_dim = config.n_embd 
        self.h_dim = config.n_embd  
        self.z_dim = config.n_embd // 2 

        # 编码器：[b*t, input_dim] => [b*t, z_dim]
        self.fc1 = nn.Linear(self.input_dim, self.h_dim) 
        self.fc2 = nn.Linear(self.h_dim, self.z_dim)  
        self.fc3 = nn.Linear(self.h_dim, self.z_dim) 

        # 解码器：[b*t, z_dim] => [b*t, input_dim]
        self.fc4 = nn.Linear(self.z_dim, self.h_dim)
        self.fc5 = nn.Linear(self.h_dim, self.input_dim)

        # 正则化
        self.resid_drop = nn.Dropout(config.resid_pdrop)

    def encode(self, x):
       
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var

    def reparameterization(self, mu, log_var):
       
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z):
        
        h = F.relu(self.fc4(z))
        x_hat = self.fc5(h) 
        return x_hat

    def forward(self, x, layer_past=None):
       
        batch_size, seq_len, embed_dim = x.shape

        #[b, t, d] => [b*t, d]
        x_flat = x.view(batch_size * seq_len, embed_dim)
        mu, log_var = self.encode(x_flat)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decode(sampled_z)

        # [b*t, d] => [b, t, d]
        x_hat = x_hat.view(batch_size, seq_len, embed_dim)
        output = self.resid_drop(x_hat)
        return output

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                             .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn =VAE(config)#  CausalSelfAttention
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 2 * config.n_embd),
            GELU(),
            nn.Linear(2 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, model_type='actor'):
        super().__init__()

        self.config = config

        self.model_type = config.model_type
        self.state_size = config.state_size

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        if model_type == 'actor':
            self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        elif model_type == 'critic':
            self.head = nn.Linear(config.n_embd, 1, bias=False)
        else:
            raise NotImplementedError

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.state_encoder = nn.Sequential(nn.Linear(self.state_size, config.n_embd), nn.Tanh())
        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.mask_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config, lr):
        """
        This long function is unfortunaely doing something very simple and is being very defensive:
        We are separating out all parameters of tthe model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif 'depth_conv.weight' in pn or 'point_conv.weight' in pn:
                    decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, pre_actions, rtgs=None, timesteps=None):
        state_embeddings = self.state_encoder(
            states.reshape(-1, self.state_size).type(torch.float32).contiguous())
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1],
                                                    self.config.n_embd)  # (batch, block_size, n_embd)

        if self.model_type == 'rtgs_state_action':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            action_embeddings = self.action_embeddings(
                pre_actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 3, self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
            num_elements = 3
        elif self.model_type == 'state_action':
            action_embeddings = self.action_embeddings(
                pre_actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2, self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings
            num_elements = 2
        elif self.model_type == 'state_only':
            token_embeddings = state_embeddings
            num_elements = 1
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)
        global_pos_emb = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1))
        global_pos_emb = torch.repeat_interleave(global_pos_emb, num_elements, dim=1)
        context_pos_emb = self.pos_emb[:, :token_embeddings.shape[1], :]
        position_embeddings = global_pos_emb + context_pos_emb

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if self.model_type == 'rtgs_state_action':
            # logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings
            logits = logits[:, 2::3, :]  # consider all tokens
        elif self.model_type == 'state_action':
            # logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
            logits = logits[:, 1::2, :]  # consider all tokens
        elif self.model_type == 'state_only':
            logits = logits
        else:
            raise NotImplementedError()

        return logits


