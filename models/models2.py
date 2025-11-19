import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn

class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-8
    @nn.compact
    def __call__(self, x):
        g = self.param('scale', nn.initializers.ones, (self.dim,))
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * (g / rms)

class SwiGLU(nn.Module):
    d_model: int
    mult: float = 2.667
    dropout: float = 0.1
    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        h = int(self.d_model * self.mult)
        u = nn.Dense(h, use_bias=False)(x)
        v = nn.Dense(h, use_bias=False)(x)
        y = nn.silu(u) * v
        y = nn.Dropout(self.dropout)(y, deterministic=deterministic)
        return nn.Dense(self.d_model, use_bias=False)(y)

class XLBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_mult: float = 2.667
    attn_dropout: float = 0.1
    mlp_dropout: float = 0.1
    resid_scale_init: float = 1e-2
    mem_len: int = 512

    @nn.compact
    def __call__(self, x, *, mem=None, deterministic: bool = True):
        B, T, D = x.shape
        mem = jnp.zeros((B, 0, D), x.dtype) if mem is None else mem
        cat = jnp.concatenate([mem, x], axis=1)
        M = mem.shape[1]

        scale = self.param('res_scale', nn.initializers.constant(self.resid_scale_init), ())

        h = RMSNorm(self.d_model)(x)
        q = nn.Dense(self.d_model, use_bias=False)(h)
        kv = nn.Dense(2 * self.d_model, use_bias=False)(RMSNorm(self.d_model)(cat))
        k, v = jnp.split(kv, 2, axis=-1)

        H = self.n_heads
        Dh = self.d_model // H
        q = q.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        k = k.reshape(B, M + T, H, Dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, M + T, H, Dh).transpose(0, 2, 1, 3)

        base = jnp.ones((1, 1, T, M + T), dtype=bool)
        tri = jnp.tril(jnp.ones((T, T), dtype=bool))
        mask = base.at[:, :, :, M:].set(tri[None, None, :, :])

        att_logits = jnp.einsum('bhtd,bhTd->bhtT', q, k) / jnp.sqrt(Dh)
        att_logits = jnp.where(mask, att_logits, jnp.finfo(att_logits.dtype).min)
        att_probs = nn.softmax(att_logits, axis=-1)
        att_probs = nn.Dropout(self.attn_dropout)(att_probs, deterministic=deterministic)
        y = jnp.einsum('bhtT,bhTd->bhtd', att_probs, v)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
        y = nn.Dense(self.d_model, use_bias=False)(y)
        x = x + scale * y

        h = RMSNorm(self.d_model)(x)
        y = SwiGLU(self.d_model, mult=self.mlp_mult, dropout=self.mlp_dropout)(h, deterministic=deterministic)
        x = x + scale * y

        new_mem = jax.lax.stop_gradient(jnp.concatenate([mem, x], axis=1))[:, -(self.mem_len):, :]
        return x, new_mem

class XLMemoryTransformer(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    mem_len: int = 512
    dropout: float = 0.1
    mlp_mult: float = 2.667

    def setup(self):
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)
        self.blocks = [XLBlock(self.d_model, self.n_heads, self.mlp_mult,
                               attn_dropout=self.dropout, mlp_dropout=self.dropout,
                               mem_len=self.mem_len) for _ in range(self.n_layers)]
        self.final_norm = RMSNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)

    def __call__(self, idx, *, mems=None, deterministic: bool = True):
        B, T = idx.shape
        x = self.tok_embed(idx)
        x = self.dropout_layer(x, deterministic=deterministic)

        if mems is None:
            mems = [None] * self.n_layers
        new_mems = []
        for blk, mem in zip(self.blocks, mems):
            x, mem_out = blk(x, mem=mem, deterministic=deterministic)
            new_mems.append(mem_out)

        x = self.final_norm(x)
        logits = jnp.einsum('btd,vd->btv', x, self.tok_embed.embedding)
        return logits, new_mems