import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

# Explicit import instead of wildcard
from .routing_weights import AtlasMultiDiffAttn


class Expert(nn.Module):
    def __init__(self, input_dim, mode, drop_ratio=0.0):
        super(Expert, self).__init__()
        mlp_hidden_dim = 2 * int(input_dim)

        self.fc = Mlp(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            out_features=(input_dim // 2 if mode != 'single' else None),
            act_layer=nn.SiLU,
            drop=drop_ratio
        )

    def forward(self, x):
        return self.fc(x)


class SharedRoutingNetwork(nn.Module):
    def __init__(self, input_dim, num_heads, num_atlas, lambda_init):
        super(SharedRoutingNetwork, self).__init__()
        self.attn = AtlasMultiDiffAttn(
            embed_dim=input_dim,
            lambda_init=lambda_init,
            num_heads=num_heads,
            num_atlas=num_atlas
        )

    def forward(self, x):
        logits = self.attn(x)
        return logits


class AtlasMoM(nn.Module):
    def __init__(self, input_dim, num_atlas, num_heads, lambda_init, mode="single"):
        super(AtlasMoM, self).__init__()

        self.mode = mode
        self.num_experts = num_atlas
        self.input_dim = input_dim
        self.lambda_init = lambda_init

        # Calculate output dimension based on mode
        out_dim = input_dim if mode == 'single' else input_dim // 2

        self.experts = nn.ModuleList([Expert(input_dim, mode, 0.3) for _ in range(num_atlas)])
        self.shared_expert = Expert(input_dim, mode, 0.1)

        self.recon_network = Mlp(
            in_features=(input_dim * 2 if mode == 'single' else input_dim),
            hidden_features=input_dim,
            out_features=input_dim,
            act_layer=nn.GELU,
            drop=0.2
        )

        self.routing_network = SharedRoutingNetwork(input_dim, num_heads, num_atlas, lambda_init)
        self.layernorm_shared = nn.LayerNorm(out_dim)
        self.layernorm_specific = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x shape: (B, N, D) | B: Batch size, N: Num atlas, D: Input dim

        # Compute routing weights
        routing_weights = self.routing_network(x)  # Shape: (B, N)
        routing_weights_expanded = routing_weights.unsqueeze(-1)  # Shape: (B, N, 1)

        # Prepare input for experts: permute to (N, B, D)
        expert_inputs = x.permute(1, 0, 2)

        # Compute expert outputs
        expert_outputs = torch.stack([
            expert(expert_inputs[i]) for i, expert in enumerate(self.experts)
        ], dim=0).permute(1, 0, 2)  # Shape: (B, N, Out_D)

        # Compute shared expert outputs
        shared_outputs = self.shared_expert(expert_inputs).permute(1, 0, 2)  # Shape: (B, N, Out_D)

        # Compute reconstruction outputs
        recon_inputs = torch.cat((expert_outputs, shared_outputs), dim=-1)
        recon_outputs = self.recon_network(recon_inputs)

        # Aggregate outputs across the atlas dimension (dim=1)
        shared_pooled = self.layernorm_shared(shared_outputs.mean(dim=1))
        expert_pooled = self.layernorm_specific((expert_outputs * routing_weights_expanded).sum(dim=1))

        final_output = torch.cat((shared_pooled, expert_pooled), dim=1)

        return final_output, expert_outputs, shared_outputs, x, recon_outputs, routing_weights