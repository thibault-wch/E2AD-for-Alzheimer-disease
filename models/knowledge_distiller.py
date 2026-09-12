import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dis_recon_loss import cross_expert_loss
from utils.tools import kd_loss


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    """
    Label smoothing implementation.
    Taken from https://github.com/MIT-HAN-LAB/ProxylessNAS/blob/master/proxyless_nas/utils.py
    """
    n_classes = pred.size(1)
    target = torch.unsqueeze(target, 1)

    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)

    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(-soft_target * F.log_softmax(pred, dim=1), dim=1))


class AtlasDistiller(nn.Module):
    def __init__(self, opt, t_net, s_net, label_smooth=False):
        super(AtlasDistiller, self).__init__()

        self.t_net = t_net
        self.s_net = s_net

        # Freeze teacher network
        self.t_net.requires_grad_(False)
        self.t_net.eval()

        self.T = 2

        self.lda_soft = opt.lda_soft
        self.lda_feat = opt.lda_feat
        self.lda_attn = opt.lda_attn
        self.lda_sh = opt.lda_sh
        self.lda_dis = opt.lda_dis
        self.lda_rec = opt.lda_rec

        t_core = t_net.module if isinstance(t_net, nn.DataParallel) else t_net
        s_core = s_net.module if isinstance(s_net, nn.DataParallel) else s_net

        self.C1 = t_core.fc.fc1.in_features
        self.C2 = s_core.fc.fc1.in_features
        if self.C1 % 2 or self.C2 % 2:
            raise ValueError(
                f'Teacher/student feature dimensions must be even, got '
                f'{self.C1} and {self.C2}'
            )

        self.W1 = nn.Parameter(torch.randn(self.C1 // 2, self.C2 // 2) * 0.02)
        self.W2 = nn.Parameter(torch.randn(self.C1 // 2, self.C2 // 2) * 0.02)

        self.softmax = nn.Softmax(dim=1)

        if label_smooth:
            self.hard_loss = cross_entropy_with_label_smoothing
        else:
            self.hard_loss = nn.CrossEntropyLoss()

    def rearrange_cross(self, A, B):
        batch, C = A.shape

        B_expanded = B.unsqueeze(0).expand(batch, *B.shape)
        temp_B = B_expanded.clone()

        mask = torch.eye(batch, device=A.device, dtype=torch.bool)
        mask = mask.unsqueeze(2).expand(batch, batch, C)

        temp_B = torch.where(mask, A.unsqueeze(0).expand(batch, batch, C), temp_B)
        return temp_B

    def gram_calculate(self, A, with_l2_norm=False):
        A = F.normalize(A, p=2, dim=-1)

        if len(A.shape) == 3:
            gram_matrix = torch.matmul(A, A.transpose(1, 2))
            if with_l2_norm:
                diagonal_sqrt = torch.sqrt(torch.diagonal(gram_matrix, offset=0, dim1=1, dim2=2))
                gram_matrix = gram_matrix / diagonal_sqrt[:, :, None]
        else:
            gram_matrix = torch.matmul(A, A.t())
            if with_l2_norm:
                diagonal_sqrt = torch.sqrt(torch.diag(gram_matrix))
                gram_matrix = gram_matrix / diagonal_sqrt[:, None]

        return gram_matrix

    def CKA(self, gram_X, gram_Y):
        # Optimized Trace Calculation: Trace of AB^T is the sum of element-wise products
        if len(gram_X.shape) == 2:
            cross_trace = (gram_X * gram_Y).sum()
            norm_X = (gram_X * gram_X).sum()
            norm_Y = (gram_Y * gram_Y).sum()
        else:
            cross_trace = (gram_X * gram_Y).sum(dim=(1, 2))
            norm_X = (gram_X * gram_X).sum(dim=(1, 2))
            norm_Y = (gram_Y * gram_Y).sum(dim=(1, 2))

        cka = 1 - (cross_trace / torch.sqrt(norm_X * norm_Y)).mean()
        return cka

    def relation_loss(self, f_t, f_s):
        with torch.no_grad():
            S_t2t = self.gram_calculate(f_t.detach())  # Teacher to Teacher

        # Project Teacher features
        f_t_transformed1 = torch.matmul(f_t[:, :self.C1 // 2], self.W1)
        f_t_transformed2 = torch.matmul(f_t[:, self.C1 // 2:], self.W2)
        f_t_proj = torch.cat((f_t_transformed1, f_t_transformed2), dim=-1)

        S_t2s = self.gram_calculate(self.rearrange_cross(f_t_proj, f_s))  # Teacher to Student

        # Project Student features
        f_s_transformed1 = torch.matmul(f_s[:, :self.C2 // 2], self.W1.t())
        f_s_transformed2 = torch.matmul(f_s[:, self.C2 // 2:], self.W2.t())
        f_s_proj = torch.cat((f_s_transformed1, f_s_transformed2), dim=-1)

        S_s2t = self.gram_calculate(self.rearrange_cross(f_s_proj, f_t))  # Student to Teacher
        S_s2s = self.gram_calculate(f_s)  # Student to Student

        # Expand Teacher-Teacher Gram matrix to match dimensions for CKA
        S_t2t_expanded_t2s = S_t2t.unsqueeze(0).expand(S_t2s.shape[0], S_t2t.shape[0], S_t2t.shape[1])
        S_t2t_expanded_s2t = S_t2t.unsqueeze(0).expand(S_s2t.shape[0], S_t2t.shape[0], S_t2t.shape[1])

        # Compute losses
        cross_s2t_loss = self.CKA(S_t2s, S_t2t_expanded_t2s)
        cross_t2s_loss = self.CKA(S_s2t, S_t2t_expanded_s2t)
        cross_s2s_loss = self.CKA(S_s2s, S_t2t)

        total_loss = cross_s2t_loss + cross_t2s_loss + cross_s2s_loss

        return total_loss, cross_t2s_loss, cross_s2t_loss, cross_s2s_loss

    def forward(self, mri_images, pet_images, atlases, label, distill_mode='multi'):
        # Forward pass through student network
        (logit_s, fea_s, expert_outputs,
         shared_outputs, expert_inputs,
         recon_outputs, ra_s) = self.s_net(mri_images, atlases)

        hard_loss = self.hard_loss(logit_s, label)

        loss_share, loss_disentanglement, loss_recon = cross_expert_loss(
            shared_outputs, expert_outputs, recon_outputs, expert_inputs
        )

        if distill_mode == 'multi':
            with torch.no_grad():
                (logit_t, fea_t, _, _, _, _, ra_t) = self.t_net(mri_images, pet_images, atlases)

            soft_loss = kd_loss(logit_s, logit_t, self.T)

            # Added 1e-8 to prevent log(0) resulting in NaN
            atlas_distill_loss = F.kl_div(torch.log(ra_s + 1e-8), ra_t.detach(), reduction='batchmean')
            rel_loss, cross_t2s_loss, cross_s2t_loss, cross_s2s_loss = self.relation_loss(fea_t, fea_s)

            # Aggregate total loss
            loss = (
                    hard_loss +
                    self.lda_sh * loss_share +
                    self.lda_dis * loss_disentanglement +
                    self.lda_rec * loss_recon +
                    self.lda_soft * soft_loss +
                    self.lda_feat * rel_loss +
                    self.lda_attn * atlas_distill_loss
            )

            return (
                logit_s, loss, hard_loss,
                self.lda_sh * loss_share,
                self.lda_dis * loss_disentanglement,
                self.lda_rec * loss_recon,
                self.lda_soft * soft_loss,
                self.lda_feat * rel_loss,
                self.lda_feat * cross_t2s_loss,
                self.lda_feat * cross_s2t_loss,
                self.lda_feat * cross_s2s_loss,
                self.lda_attn * atlas_distill_loss
            )

        else:
            loss = (
                    hard_loss +
                    self.lda_sh * loss_share +
                    self.lda_dis * loss_disentanglement +
                    self.lda_rec * loss_recon
            )

            return (
                logit_s, loss, hard_loss,
                self.lda_sh * loss_share,
                self.lda_dis * loss_disentanglement,
                self.lda_rec * loss_recon
            )