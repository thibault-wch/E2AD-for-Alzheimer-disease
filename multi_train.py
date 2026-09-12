import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import wandb

from options.train_options import TrainOptions
from utils.multi_train_cnad import train_cnad
from utils.tools import get_scheduler, save_results_to_csv, define_Cls
from utils.OurDataset import OurDataset
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

if __name__ == '__main__':
    # Initialize configurations
    opt = TrainOptions().parse()

    # Initialize metric containers
    metrics = {
        'loss': [], 'acc': [], 'f1': [],
        'recall': [], 'spe': [], 'auc': [], 'precision': []
    }

    # Setup device dynamically to avoid multi-GPU conflicts
    device = torch.device('cuda:0' if opt.gpu_ids and torch.cuda.is_available() else 'cpu')

    # 5-fold cross-validation loop
    for item_fold in range(5):
        # Initialize wandb logging
        wandb.init(
            project="Final_CNAD",
            name=f"{opt.name}_{item_fold}",
            config={
                "group": opt.group,
                "learning_rate": opt.lr,
                "architecture": opt.cls_type,
                "epoch": opt.epoch_count,
            }
        )

        # Dynamic path formatting
        pretrained_path = (
            os.path.join(opt.pretrained_multi, f'{item_fold}_net.pth')
            if opt.pretrained_multi else None
        )
        # Build model and environment
        criterion = nn.CrossEntropyLoss()
        model = define_Cls(
            opt.cls_type,
            class_num=opt.class_num,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            pretrained=pretrained_path,
            gpu_ids=opt.gpu_ids
        )
        torch.cuda.empty_cache()

        # Optimizer and Scheduler
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.95), lr=opt.lr)
        scheduler = get_scheduler(optimizer, opt)
        loss_scaler = NativeScaler()

        # Dataloaders
        train_set = OurDataset(mode='train', root=opt.root, type='multi_pet', mtype='NCAD', fold=item_fold)
        print(f"Length of train list (Fold {item_fold}): {len(train_set)}")
        train_loader = DataLoader(
            train_set,
            batch_size=opt.batch_size,
            num_workers=opt.workers,
            shuffle=True
        )

        test_set = OurDataset(mode="test", root=opt.root, type='multi_pet', mtype='NCAD', fold=item_fold)
        test_loader = DataLoader(
            test_set,
            batch_size=opt.batch_size,
            num_workers=opt.workers,
            drop_last=False,
            shuffle=False
        )

        # Directory management
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)

        # Execute training pipeline
        results = train_cnad(
            item_fold, model, opt, train_loader, test_loader, opt.epoch_count,
            optimizer, scheduler, criterion, expr_dir, opt.print_freq,
            opt.save_epoch_freq, loss_scaler, opt.accum_iter, device
        )

        # Unpack metrics
        (train_loss, val_loss, train_acc, val_acc, train_f1_score, val_f1_score,
         train_recall, val_recall, train_spe, val_spe, train_auc, val_auc,
         val_precision, train_precision) = results

        # Append metrics for current fold
        metrics['acc'].append(val_acc)
        metrics['loss'].append(val_loss)
        metrics['f1'].append(val_f1_score)
        metrics['auc'].append(val_auc)
        metrics['spe'].append(val_spe)
        metrics['precision'].append(val_precision)
        metrics['recall'].append(val_recall)

        wandb.finish()

    # Aggregate and save results
    save_path = os.path.join(opt.checkpoints_dir, opt.name)
    save_results_to_csv(
        metrics['acc'], metrics['loss'], metrics['f1'],
        metrics['auc'], metrics['spe'], metrics['precision'],
        metrics['recall'], save_path
    )