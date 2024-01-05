import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix


def spe(confusion_matrix):

    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    return np.average(TN/(TN+FP)), np.average(TP/(TP+FP))


def train_epoch(model, loader, device, train_optimizer, lossfunction):

    model.train()
    total_loss, total_num = 0.0, 0
    pred_list, target_list = [], []

    for batch_idx, (fc_matrixs, targets, posi) in enumerate(loader):

        batch_size = fc_matrixs.shape[0]
        fc_matrixs = fc_matrixs.to(device)
        posi = posi.to(device)
        targets = targets.to(device)
        targets = torch.as_tensor(targets, dtype=torch.long).to(device)

        out_1 = model(fc_matrixs, posi)

        loss = lossfunction(out_1, targets)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        preds_score = F.softmax(out_1, dim=1).detach().cpu().numpy()
        preds = np.argmax(preds_score, axis=1)
        targets = targets.cpu().numpy()

        pred_list.append(preds)
        target_list.append(targets)

    preds = np.concatenate(pred_list)
    targets = np.concatenate(target_list)

    score_acc = accuracy_score(targets, preds)

    return total_loss / total_num, score_acc


def val_epoch(model, loader, device, lossfunction):

    model.eval()
    total_loss, total_num = 0.0, 0
    pred_list, target_list = [], []
    preds_score_list = []

    with torch.no_grad():
        for batch_idx, (fc_matrixs, targets, posi) in enumerate(loader):

            batch_size = fc_matrixs.shape[0]
            fc_matrixs = fc_matrixs.to(device)
            posi = posi.to(device)
            targets = torch.as_tensor(targets, dtype=torch.long).to(device)

            out_1 = model(fc_matrixs, posi)

            loss = lossfunction(out_1, targets)

            preds_score = F.softmax(out_1, dim=1).cpu().numpy()
            preds = np.argmax(preds_score, axis=1)
            targets = targets.cpu().numpy()
            pred_list.append(preds)
            target_list.append(targets)
            preds_score_list.append(preds_score[:, 1])

            total_num += batch_size
            total_loss += loss.item() * batch_size

        preds = np.concatenate(pred_list)
        preds_score = np.concatenate(preds_score_list)
        targets = np.concatenate(target_list)

        f1 = f1_score(targets, preds)
        precision = precision_score(targets, preds)
        recall = recall_score(targets, preds)
        specificity, test = spe(confusion_matrix(targets, preds))
        score_acc = accuracy_score(targets, preds)
        score_auc = roc_auc_score(targets, preds_score)

        return total_loss / total_num, score_acc, score_auc, f1, precision, recall, specificity



