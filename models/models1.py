import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn

class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-8
    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.ones, (self.dim,))
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return x * (scale / rms)

class SwiGLU(nn.Module):
    d_model: int
    mult: float = 2.667
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        hidden = int(self.d_model * self.mult)
        u = nn.Dense(hidden, use_bias=False)(x)
        v = nn.Dense(hidden, use_bias=False)(x)
        x = nn.silu(u) * v
        x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(self.d_model, use_bias=False)(x)
        return x

def rotary_emb(d: int, T: int, base: float = 10000.0, dtype=jnp.float32):
    inv_freq = 1.0 / (base ** (jnp.arange(0, d, 2, dtype=dtype) / d))
    t = jnp.arange(T, dtype=dtype)
    freqs = jnp.einsum('t,f->tf', t, inv_freq)
    cos, sin = jnp.cos(freqs), jnp.sin(freqs)
    return cos[None, None, ...], sin[None, None, ...]

def apply_rope(x, cos, sin):
    x1, x2 = jnp.split(x, 2, axis=-1)
    xc = x1 * cos - x2 * sin
    xs = x1 * sin + x2 * cos
    return jnp.concatenate([xc, xs], axis=-1)


class MQSelfAttention(nn.Module):
    d_model: int
    n_heads: int
    n_kv_heads: int = 1
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, *, mask=None, deterministic: bool = True):
        H = self.n_heads
        H_kv = self.n_kv_heads
        Dh = self.d_model // H

        q = nn.Dense(self.d_model, use_bias=False)(x)
        k = nn.Dense(H_kv * Dh, use_bias=False)(x)
        v = nn.Dense(H_kv * Dh, use_bias=False)(x)

        B, T, _ = q.shape
        q = q.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H_kv, Dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H_kv, Dh).transpose(0, 2, 1, 3)

        cos, sin = rotary_emb(Dh, T)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if H_kv != H:
            factor = H // H_kv
            k = jnp.repeat(k, repeats=factor, axis=1)
            v = jnp.repeat(v, repeats=factor, axis=1)

        scale = (1.0 / jnp.sqrt(jnp.array(Dh, dtype=x.dtype)))
        att = jnp.einsum('bhtd,bhTd->bhtT', q, k) * scale

        if mask is not None:
            att = jnp.where(mask, att, jnp.finfo(att.dtype).min)
        att = nn.softmax(att, axis=-1)
        att = nn.Dropout(self.dropout)(att, deterministic=deterministic)

        y = jnp.einsum('bhtT,bhTd->bhtd', att, v)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        y = nn.Dense(self.d_model, use_bias=False)(y)
        y = nn.Dropout(self.dropout)(y, deterministic=deterministic)
        return y

class DecoderBlock(nn.Module):
    d_model: int; n_heads: int; n_kv_heads: int = 1
    attn_dropout: float = 0.1
    mlp_dropout: float = 0.1
    resid_scale_init: float = 1e-2

    @nn.compact
    def __call__(self, x, *, mask=None, deterministic: bool = True):
        scale = self.param('res_scale', nn.initializers.constant(self.resid_scale_init), ())

        h = RMSNorm(self.d_model)(x)
        h = MQSelfAttention(self.d_model, self.n_heads, self.n_kv_heads, dropout=self.attn_dropout)(
            h, mask=mask, deterministic=deterministic)
        x = x + scale * h

        h = RMSNorm(self.d_model)(x)
        h = SwiGLU(self.d_model, mult=2.667, dropout=self.mlp_dropout)(h, deterministic=deterministic)
        x = x + scale * h
        return x

class DecoderOnlyTransformer(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int = 1
    max_len: int = 2048
    dropout: float = 0.1

    def setup(self):
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)
        self.blocks = [
            DecoderBlock(
                self.d_model,
                self.n_heads,
                self.n_kv_heads,
                attn_dropout=self.dropout,
                mlp_dropout=self.dropout,
            )
            for _ in range(self.n_layers)
        ]
        self.final_norm = RMSNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)

    def __call__(self, idx, *, deterministic: bool = True, pad_mask: jnp.ndarray | None = None):
        B, T = idx.shape
        x = self.tok_embed(idx)
        x = self.dropout_layer(x, deterministic=deterministic)

        causal = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))
        mask = causal if pad_mask is None else attn.combine_masks(
            causal, attn.make_attention_mask(pad_mask, pad_mask, dtype=bool)
        )

        for blk in self.blocks:
            x = blk(x, mask=mask, deterministic=deterministic)

        x = self.final_norm(x)
        logits = jnp.einsum('btd,vd->btv', x, self.tok_embed.embedding)
        return logits
