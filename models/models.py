import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn

class MLP(nn.Module):
    d_model: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x):
        hidden = int(self.d_model * self.mlp_ratio)
        x = nn.Dense(hidden)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model)(x)
        return x

class DecoderBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x, *, mask=None):
        # attention
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            use_bias=False,
        )(h, mask=mask)
        x = x + h

        # mlp
        h = nn.LayerNorm()(x)
        h = MLP(self.d_model, mlp_ratio=self.mlp_ratio)(h)
        x = x + h
        return x

class DecoderOnlyTransformer(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    mlp_ratio: int = 4

    def setup(self):
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)
        self.positional_embed = self.param(
            "positional_embed",
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model),
        )

        blocks = []
        for i in range(self.n_layers):
            # 第一层轻一点，其它层用默认的
            ratio = 2 if i == 0 else self.mlp_ratio
            blocks.append(
                DecoderBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    mlp_ratio=ratio,
                )
            )
        self.blocks = blocks

        self.layerNorm_final = nn.LayerNorm()
        # 留着，万一你想切回 Dense 头
        self.project_to_vocab = nn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok_embed(idx) + self.positional_embed[:T]
        causal = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))
        for blk in self.blocks:
            x = blk(x, mask=causal)
        x = self.layerNorm_final(x)

        # 真正的权重共享版本：
        E = self.tok_embed.embedding  # (V, D)
        logits = jnp.einsum("btd,vd->btv", x, E)

        # 如果你想对比，也可以用这一句：
        # logits = self.project_to_vocab(x)

        return logits
