import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, matthews_corrcoef, \
    average_precision_score
import wandb

# Ensure explicit relative or absolute imports based on your project structure
from .OurDataset import OurDataset
from .dis_recon_loss import cross_expert_loss
from .tools import balanced_accuracy


def train_cnad(fold, model, opt, train_loader, test_loader, epochs, optimizer, scheduler, criterion,
               expr_dir, print_freq, save_epoch_freq, loss_scaler, accum_iter, device='cpu'):
    """
    Executes the training and validation loop for the multi-modal network.
    """
    steps = 0
    start = time.time()
    best_auc = 0.0

    for e in tqdm(range(1, epochs + 1), desc=f"Fold {fold} Training"):
        model.train()

        # Clean slate for gradient accumulation at the start of epoch
        optimizer.zero_grad()

        train_loss = 0.0
        train_correct_sum = 0.0
        train_simple_cnt = 0.0

        y_train_true = []
        y_train_pred = []
        train_prob_all = []
        train_label_all = []

        for ii, item in enumerate(tqdm(train_loader, desc=f"Epoch {e}/{epochs}", leave=False)):
            steps += 1

            # Data loading & device allocation
            mri_images = item['mri_x_lb'].to(device)
            pet_images = item['pet_x_lb'].to(device)
            atlases = item['atlas_x_lb'].to(device)
            labels = item['y_lb'].to(device)

            # Forward pass
            outputs, _, expert_outputs, shared_outputs, expert_inputs, recon_outputs, _ = model(mri_images, pet_images,
                                                                                                atlases)

            # Loss computation
            loss_cls = criterion(outputs, labels)
            loss_share, loss_distenglement, loss_recon = cross_expert_loss(
                shared_outputs, expert_outputs, recon_outputs, expert_inputs
            )

            loss = loss_cls + opt.lda_sh * loss_share + opt.lda_dis * loss_distenglement + opt.lda_rec * loss_recon
            loss = loss / accum_iter

            update_grad = ((ii + 1) % accum_iter == 0 or (ii + 1) == len(train_loader))

            loss_scaler(loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=update_grad,)

            if update_grad:
                optimizer.zero_grad()
            train_loss += loss.item() * accum_iter  # Re-scale for accurate logging

            # Step-level WandB logging
            wandb.log({
                'lr': optimizer.param_groups[0]['lr'],
                'cls_loss': loss_cls.item(),
                'share_loss': loss_share.item(),
                'distenglement_loss': loss_distenglement.item(),
                'recon_loss': loss_recon.item(),
                'step_train_loss': loss.item() * accum_iter
            })

            # Metrics aggregation
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct_sum += (labels.data == train_predicted).sum().item()
            train_simple_cnt += labels.size(0)

            y_train_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
            y_train_pred.extend(np.ravel(np.squeeze(train_predicted.cpu().detach().numpy())).tolist())

            outputs_softmax = outputs.softmax(dim=-1)
            train_prob_all.extend(outputs_softmax[:, 1].cpu().detach().numpy())
            train_label_all.extend(labels.cpu().numpy())  # [FIX] Converted to numpy to prevent memory leak

        if scheduler:
            scheduler.step()

        # Validation Phase
        val_correct_sum = 0.0
        val_simple_cnt = 0.0
        val_loss = 0.0

        y_val_true = []
        y_val_pred = []
        val_prob_all = []
        val_label_all = []

        model.eval()
        with torch.no_grad():
            for ii, item in enumerate(test_loader):
                mri_images = item['mri_x_lb'].to(device)
                pet_images = item['pet_x_lb'].to(device)
                atlases = item['atlas_x_lb'].to(device)
                labels = item['y_lb'].to(device)

                outputs, _, _, _, _, _, _ = model.forward(mri_images, pet_images, atlases)

                loss_cls = criterion(outputs, labels).item()
                val_loss += loss_cls

                _, val_predicted = torch.max(outputs.data, 1)
                val_correct_sum += (labels.data == val_predicted).sum().item()
                val_simple_cnt += labels.size(0)

                y_val_true.extend(np.ravel(np.squeeze(labels.cpu().numpy())).tolist())
                y_val_pred.extend(np.ravel(np.squeeze(val_predicted.cpu().numpy())).tolist())

                outputs_softmax = outputs.softmax(dim=-1)
                val_prob_all.extend(outputs_softmax[:, 1].cpu().numpy())
                val_label_all.extend(labels.cpu().numpy())

        # Calculate final metrics for the epoch
        val_loss = val_loss / len(test_loader)
        train_loss = train_loss / len(train_loader)

        # Validation metrics
        val_mcc = matthews_corrcoef(y_val_true, y_val_pred)
        val_acc = val_correct_sum / val_simple_cnt
        val_bac = balanced_accuracy(y_val_true, y_val_pred)
        val_auc = roc_auc_score(val_label_all, val_prob_all, average='weighted')
        val_f1_score = f1_score(y_val_true, y_val_pred, average='weighted')
        val_recall = recall_score(y_val_true, y_val_pred, average='weighted')
        val_spe = recall_score(y_val_true, y_val_pred, pos_label=0, average='binary')
        val_precision = precision_score(y_val_true, y_val_pred, average='weighted', zero_division=0)
        val_ap = average_precision_score(val_label_all, val_prob_all, average='weighted')

        # Train metrics
        train_mcc = matthews_corrcoef(y_train_true, y_train_pred)
        train_acc = train_correct_sum / train_simple_cnt
        train_bac = balanced_accuracy(y_train_true, y_train_pred)
        train_f1_score = f1_score(y_train_true, y_train_pred, average='weighted')
        train_recall = recall_score(y_train_true, y_train_pred, average='weighted')
        train_spe = recall_score(y_train_true, y_train_pred, pos_label=0, average='binary')
        train_auc = roc_auc_score(train_label_all, train_prob_all, average='weighted')
        train_precision = precision_score(y_train_true, y_train_pred, average='weighted', zero_division=0)
        train_ap = average_precision_score(train_label_all, train_prob_all, average='weighted')

        # Logging and Console Output
        if e % print_freq == 0:
            wandb.log({
                "epoch": e,
                "train_acc": train_acc, "train_f1": train_f1_score,
                "train_bac": train_bac, "train_mcc": train_mcc,
                "train_ap": train_ap, "train_sen": train_recall,
                "train_spe": train_spe, "train_auc": train_auc,
                "val_loss": val_loss, "val_acc": val_acc,
                "val_auc": val_auc, "val_f1": val_f1_score,
                "val_bac": val_bac, "val_ap": val_ap,
                "val_mcc": val_mcc, "val_pre": val_precision,
                "val_sen": val_recall, "val_spe": val_spe
            })

            print(f"\nEpoch: {e}/{epochs}")
            print(
                f"  [Train] Loss: {train_loss:.3f} | Acc: {train_acc:.3f} | AUC: {train_auc:.3f} | F1: {train_f1_score:.3f} | Pre: {train_precision:.3f} | SEN: {train_recall:.3f} | SPE: {train_spe:.3f} | BAC: {train_bac:.3f} | AP: {train_ap:.3f} | MCC: {train_mcc:.3f}")
            print(
                f"  [Val]   Loss: {val_loss:.3f} | Acc: {val_acc:.3f} | AUC: {val_auc:.3f} | F1: {val_f1_score:.3f} | Pre: {val_precision:.3f} | SEN: {val_recall:.3f} | SPE: {val_spe:.3f} | BAC: {val_bac:.3f} | AP: {val_ap:.3f} | MCC: {val_mcc:.3f}\n")

        # Checkpointing
        if e % save_epoch_freq == 0:
            epoch_model_path = os.path.join(expr_dir, f"{e}_{fold}_net.pth")
            torch.save(model.state_dict(), epoch_model_path)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_path = os.path.join(expr_dir, f"{fold}_best_net.pth")
            torch.save(model.state_dict(), best_model_path)

    # Final timing log
    end = time.time()
    running_time = end - start
    print(f"Training completed in {running_time // 60:.0f}m {running_time % 60:.0f}s")

    # Return final epoch metrics
    return (train_loss, val_loss, train_acc, val_acc, train_f1_score, val_f1_score,
            train_recall, val_recall, train_spe, val_spe, train_auc, val_auc,
            val_precision, train_precision)