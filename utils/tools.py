import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import init
from torch.optim import lr_scheduler
from sklearn.metrics import balanced_accuracy_score

# Ensure these relative/absolute imports match your project structure
from models.AtlasNet import AtlasNet, BasicBlock
from models.MultiAtlasNet import MultiAtlasNet


def kd_loss(student_logits, teacher_logits, temperature=2.0):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean',
    ) * (temperature ** 2)

def balanced_accuracy(y_true, y_pred):
    """
    Computes balanced accuracy safely for both binary and multiclass inputs.
    Replaces brittle confusion_matrix.ravel() unpacking.
    """
    return balanced_accuracy_score(y_true, y_pred)


def save_results_to_csv(val_acc_list, val_loss_list, val_f1_score_list,
                        val_auc_list, val_spe_list, val_precision_list,
                        val_recall_list, result_dir):
    """Aggregates cross-validation fold metrics and saves to CSV."""
    num_folds = len(val_acc_list)
    assert num_folds == len(val_loss_list) == len(val_f1_score_list) == len(val_auc_list) \
           == len(val_spe_list) == len(val_precision_list) == len(val_recall_list), \
        "All metric lists must have the same length."

    os.makedirs(result_dir, exist_ok=True)

    results_df = pd.DataFrame({
        'fold': [f'fold_{i}' for i in range(num_folds)],
        'val_acc': val_acc_list,
        'val_loss': val_loss_list,
        'val_f1_score': val_f1_score_list,
        'val_auc': val_auc_list,
        'val_spe': val_spe_list,
        'val_precision': val_precision_list,
        'val_recall': val_recall_list
    })

    # Save raw fold results
    results_path = os.path.join(result_dir, 'results.csv')
    results_df.to_csv(results_path, index=False)

    # Calculate and save summary statistics
    summary_df = results_df.describe().transpose()[['mean', 'std']].reset_index()
    summary_df = summary_df.rename(columns={'index': 'metric'})

    summary_path = os.path.join(result_dir, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print("\n[Cross-Validation Summary]")
    print(summary_df)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup."""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epoch_count - args.warmup_epochs)))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_scheduler(optimizer, opt):
    """Defines and returns the learning rate scheduler based on options."""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'lambda_exp':
        def lambda_rule(epoch):
            if epoch < opt.warmup_epochs:
                return min(1.0, (epoch + 1) / opt.warmup_epochs)  # Warmup
            return max(0.02, 1.0 * (opt.lr_decay ** (epoch - opt.warmup_epochs)))  # Exponential decay

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'lambda_cosine':
        def lambda_rule(epoch):
            if epoch < opt.warmup_epochs:
                return epoch / opt.warmup_epochs
            return max(1e-5, 0.5 * (
                        1. + math.cos(math.pi * (epoch - opt.warmup_epochs) / (opt.epoch_count - opt.warmup_epochs))))

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch_count, eta_min=0)
    elif opt.lr_policy == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)
    else:
        raise NotImplementedError(f"Learning rate policy [{opt.lr_policy}] is not implemented")

    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    """Initializes network weights dynamically."""

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f"Initialization method [{init_type}] is not implemented")

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print(f"Initializing network with: {init_type}")
    net.apply(init_func)


def cudanet(net, gpu_ids=None):
    """Safely mounts the network to the specified GPUs."""
    if gpu_ids is None:
        gpu_ids = []

    if len(gpu_ids) > 0 and torch.cuda.is_available():
        # Ensure model maps to the primary target GPU first
        device = torch.device(f"cuda:{gpu_ids[0]}")
        net.to(device)

        # Apply DataParallel if multiple GPUs are specified
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, device_ids=gpu_ids)

    return net


def load_checkpoint(net, pretrained=None):
    if pretrained is None:
        return net
    if not os.path.isfile(pretrained):
        raise FileNotFoundError(f'Checkpoint not found: {pretrained}')

    print(f'Loading pretrained model from: {pretrained}')
    state_dict = torch.load(pretrained, map_location='cpu')
    net.load_state_dict(state_dict, strict=True)
    return net


def define_Cls(netCls, class_num=4, lambda_init=0.6, init_type='normal',
               init_gain=0.02, pretrained=None, gpu_ids=None):
    """Factory function to build, initialize, and mount the classification model."""
    if gpu_ids is None:
        gpu_ids = []

    if netCls == 'multi_atlas':
        net = MultiAtlasNet(
            BasicBlock, [3, 3], shortcut_type='B',
            num_classes=class_num, num_heads=8, dim_latent=256,
            num_atlas=56, lambda_init=lambda_init
        )
    elif netCls == 'single_atlas':
        net = AtlasNet(
            BasicBlock, [3, 3], shortcut_type='B',
            num_classes=class_num, num_heads=8, dim_latent=256,
            num_atlas=56, lambda_init=lambda_init
        )
    else:
        raise ValueError(f"Unknown network architecture: {netCls}")

    init_weights(net, init_type, gain=init_gain)
    net = load_checkpoint(net, pretrained)
    return cudanet(net, gpu_ids)