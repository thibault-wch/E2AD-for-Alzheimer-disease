import torch
import torch.nn.functional as F
from torch import nn


class AtlasMultiDiffAttn(nn.Module):
    def __init__(
            self,
            embed_dim=128,
            lambda_init=0.7,
            num_atlas=56,
            num_heads=4,
            qk_norm=True,
            norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()

        # Ensure dimensional requirements are met
        assert embed_dim % (num_heads * 2) == 0, "embed_dim must be divisible by 2 * num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_atlas = num_atlas

        self.head_dim = embed_dim // (num_heads * 2)
        self.scaling = self.head_dim ** -0.5

        self.q_activation = nn.SiLU()

        # q_proj_emb treats sequence length (num_atlas) as channels
        self.q_proj_emb = nn.Conv1d(
            in_channels=num_atlas,
            out_channels=num_atlas,
            kernel_size=7,
            padding=3
        )

        # q_proj_atlas treats embed_dim as channels
        self.q_proj_atlas = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=7,
            padding=3
        )

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.lambda_init = lambda_init

        # Idiomatic parameter initialization
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim, dtype=torch.float32) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim, dtype=torch.float32) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim, dtype=torch.float32) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim, dtype=torch.float32) * 0.1)

    def forward(self, x):
        bsz, tgt_len, embed_dim = x.size()  # Shape: (B, L, C)

        # Ensure sequence length matches the configured atlas count for Conv1d
        assert tgt_len == self.num_atlas, f"Expected sequence length {self.num_atlas}, got {tgt_len}"

        # -----------------------------------------
        #  Process Query (q) - Unrolled for clarity
        # -----------------------------------------
        # Conv over embedding dim. Input shape: (B, L, C) -> L is treated as channels
        q_emb = self.q_proj_emb(x)
        q_emb = self.q_activation(q_emb)

        # Permute for second Conv. Shape becomes: (B, C, L) -> C is treated as channels
        q_atlas_in = q_emb.permute(0, 2, 1)
        q_atlas_out = self.q_proj_atlas(q_atlas_in)
        q_atlas_out = self.q_activation(q_atlas_out)

        # Pool over the sequence (atlas) dimension and permute back
        q_pooled = torch.mean(q_atlas_out, dim=2, keepdim=True)  # Shape: (B, C, 1)
        q = q_pooled.permute(0, 2, 1)  # Shape: (B, 1, C)

        # -----------------------------------------
        #  Process Key (k)
        # -----------------------------------------
        k = self.k_proj(x)  # Shape: (B, L, C)

        # -----------------------------------------
        #  Multi-Head Reshaping & Normalization
        # -----------------------------------------
        # Reshape to (B, L, 2*H, Head_D)
        q = q.view(bsz, 1, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scaling

        # -----------------------------------------
        #  Attention Computation
        # -----------------------------------------
        # einsum maps: b(batch), q(query len=1), h(heads*2), e(head_dim) -> (B, 2*H, 1, L)
        attn_weights = torch.einsum("bqhe,bkhe->bhqk", q, k)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # Compute differential lambdas
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).to(q.dtype)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).to(q.dtype)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Reshape to separate the differential pairs: (B, H, 2, 1, L)
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, 1, tgt_len)

        # Differential attention logic
        attn_diff = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        # Softmax, aggregate across heads, and flatten cleanly to (B, L)
        attn_weights = torch.softmax(attn_diff, dim=-1).mean(dim=1).flatten(start_dim=1)

        return attn_weights