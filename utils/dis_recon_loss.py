import torch
import torch.nn.functional as F


def batch_cosine_similarity(x, y):
    """
    Computes pairwise cosine similarity for a batch of feature sets.
    Returns shape: (B, J, J)
    """
    x_normalized = F.normalize(x, p=2, dim=-1)
    y_normalized = F.normalize(y, p=2, dim=-1)
    # Matmul with transposed y to get pairwise dot products (similarities)
    cos_sim = torch.matmul(x_normalized, y_normalized.transpose(1, 2))
    return cos_sim


def double_center(dist_matrix):
    """
    Applies double centering to a batch of matrices.
    """
    # dist_matrix shape: (B, J, J)
    row_mean = dist_matrix.mean(dim=2, keepdim=True)  # (B, J, 1)
    col_mean = dist_matrix.mean(dim=1, keepdim=True)  # (B, 1, J)
    global_mean = dist_matrix.mean(dim=[1, 2], keepdim=True)  # (B, 1, 1)

    # Double-centered matrix
    centerized_matrix = dist_matrix - row_mean - col_mean + global_mean
    return centerized_matrix


def share_loss(shared_features):
    """
    Encourages shared features to align with their global mean (consensus).
    """
    mean_shared = shared_features.mean(dim=1, keepdim=True)  # (B, 1, C)
    cos_sim = F.cosine_similarity(shared_features, mean_shared, dim=-1)  # (B, J)

    # Convert similarity to distance and average
    cos_distance = 1.0 - cos_sim
    return cos_distance.mean()


def disentanglement_loss(shared_features, expert_features):
    """
    Penalizes correlations between different experts (off-diagonal)
    and between shared/expert features (diagonal).
    """
    B, J, _ = expert_features.size()

    # Create mask on the correct device
    mask = torch.eye(J, device=expert_features.device).expand(B, J, J)

    # [1] Separation for specific experts (Penalize off-diagonal similarities)
    sim_experts = batch_cosine_similarity(expert_features, expert_features)
    sim_experts_centered = double_center(sim_experts)

    # Account in the denominator
    num_off_diagonal_elements =  J * (J - 1)
    loss_expert_indep = (torch.abs(sim_experts_centered) * (1 - mask)).sum() / num_off_diagonal_elements

    # [2] Decorrelation for shared vs specific experts (Penalize diagonal similarities)
    sim_shared_expert = batch_cosine_similarity(shared_features, expert_features)
    sim_shared_expert_centered = double_center(sim_shared_expert)

    # Account for Batch dimension (B) in the denominator
    num_diagonal_elements = J
    loss_shared_indep = (torch.abs(sim_shared_expert_centered) * mask).sum() / num_diagonal_elements

    return loss_expert_indep + loss_shared_indep


def cross_expert_loss(shared_outputs, expert_outputs, recon_outputs, expert_inputs):
    """
    Aggregates sharing, disentanglement, and reconstruction losses.
    """
    loss_share = share_loss(shared_outputs)
    loss_disentangle = disentanglement_loss(shared_outputs, expert_outputs)
    loss_recon = F.mse_loss(recon_outputs, expert_inputs)

    return loss_share, loss_disentangle, loss_recon