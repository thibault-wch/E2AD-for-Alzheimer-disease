import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import wandb

from models.knowledge_distiller import AtlasDistiller
from options.train_options import TrainOptions
from utils.distill_train_cnad import train_cnad
from utils.tools import get_scheduler, save_results_to_csv, define_Cls, cudanet
from utils.OurDataset import OurDataset
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

if __name__ == '__main__':
    # Initialize configurations
    opt = TrainOptions().parse()

    # Setup device dynamically to avoid multi-GPU assignment conflicts
    device = torch.device('cuda:0' if opt.gpu_ids and torch.cuda.is_available() else 'cpu')

    # Centralize metric tracking
    metrics = {
        'loss': [], 'acc': [], 'f1': [],
        'recall': [], 'spe': [], 'auc': [], 'precision': []
    }

    # Execute 5-fold cross-validation
    for fold in range(5):
        # Initialize Weights & Biases
        wandb.init(
            project="Distill_CNAD",
            name=f"{opt.name}_{fold}",
            config={
                "learning_rate": opt.lr,
                "epoch": opt.epoch_count,
            }
        )

        criterion = nn.CrossEntropyLoss()
        if not opt.pretrained_multi:
            raise ValueError('--pretrained_multi is required for distillation')
        # Format pretrained paths safely
        multi_pretrained = os.path.join(opt.pretrained_multi, f'50_{fold}_net.pth') if opt.pretrained_multi else None
        single_pretrained = os.path.join(opt.pretrained_single,
                                         f'40_{fold}_net.pth') if opt.pretrained_single else None

        # Initialize Teacher (Multi-Atlas) and Student (Single-Atlas) models
        multi_model = define_Cls(
            'multi_atlas', class_num=opt.class_num,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            pretrained=multi_pretrained,
            gpu_ids=opt.gpu_ids
        )

        single_model = define_Cls(
            'single_atlas', class_num=opt.class_num,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            pretrained=single_pretrained,
            gpu_ids=opt.gpu_ids
        )

        # Wrap models in distiller framework
        distill_model = cudanet(
            AtlasDistiller(t_net=multi_model, s_net=single_model, opt=opt),
            opt.gpu_ids
        )

        torch.cuda.empty_cache()

        # Optimizer targets only the student network's parameters
        trainable_params = [p for p in distill_model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, betas=(0.9, 0.95), lr=opt.lr)
        scheduler = get_scheduler(optimizer, opt)
        loss_scaler = NativeScaler()

        # [1] Paired train set (PET)
        train_set_pet = OurDataset(mode='train', root=opt.root, type='multi_pet', mtype='NCAD', fold=fold)
        print(f"Length train[paired] list: {len(train_set_pet)}")
        train_loader_pet = DataLoader(
            train_set_pet, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=True
        )

        # [2] Single train set (MRI)
        train_set_mri = OurDataset(mode='train', root=opt.root, type='multi_mri', mtype='NCAD', fold=fold)
        print(f"Length train[single] list: {len(train_set_mri)}")
        train_loader_mri = DataLoader(
            train_set_mri, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=True
        )

        # [3] Single test set
        test_set = OurDataset(mode="test", root=opt.root, type='single', mtype='NCAD', fold=fold)
        test_loader = DataLoader(
            test_set, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=False
        )

        # Directory management
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)

        # Execute training pipeline (Fixed missing train_loss unpack variable)
        results = train_cnad(
            fold, distill_model, train_loader_pet, train_loader_mri, test_loader, opt.epoch_count,trainable_params,
            optimizer, scheduler, criterion, expr_dir, opt.print_freq,
            opt.save_epoch_freq, loss_scaler, opt.accum_iter, device
        )



        # Unpack metrics safely
        (train_loss, val_loss, train_acc, val_acc, train_f1_score, val_f1_score,
         train_recall, val_recall, train_spe, val_spe, train_auc, val_auc,
         val_precision, train_precision) = results

        # Record validation metrics for current fold
        metrics['loss'].append(val_loss)
        metrics['acc'].append(val_acc)
        metrics['f1'].append(val_f1_score)
        metrics['recall'].append(val_recall)
        metrics['spe'].append(val_spe)
        metrics['auc'].append(val_auc)
        metrics['precision'].append(val_precision)

        wandb.finish()

    # Save aggregated results
    save_path = os.path.join(opt.checkpoints_dir, opt.name)
    save_results_to_csv(
        metrics['acc'], metrics['loss'], metrics['f1'],
        metrics['auc'], metrics['spe'], metrics['precision'],
        metrics['recall'], save_path
    )