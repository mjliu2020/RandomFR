import time, os, argparse, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime
import warnings
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")

from Code.NYU_dataset import get_data_loader
from Code.NYU_net_utils import train_epoch, val_epoch
from Code.model.NYU_ViT import ViT


def open_log(log_path, name='train'):
    log_savepath = os.path.join(log_path, name)
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    log_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if os.path.isfile(os.path.join(log_savepath, '{}.log'.format(log_name))):
        os.remove(os.path.join(log_savepath, '{}.log'.format(log_name)))
    initLogging(os.path.join(log_savepath, '{}.log'.format(log_name)))


# Init for logging
def initLogging(logFilename):
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s-%(levelname)s] %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S',
                        filename=logFilename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def train(i_fold, DEVICE):

    weight_decay = 0.0001
    lr = 9e-4
    lr_decay = 9e-4
    train_epochs = 100

    patience = 100
    cnt_wait = 0
    best_val_acc = 1e-9

    network_name = f'lr{lr:.1e}{lr_decay:.1e}_wd{weight_decay}_fold{i_fold}'
    save_path = f'result/NYU_PatchSize8/test1/{network_name}'
    code_path = 'result/NYU_PatchSize8/test1/code'
    os.makedirs(code_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    shutil.copy('./NYU_train.py', code_path)
    shutil.copy('./NYU_Dataset.py', code_path)
    shutil.copy('./NYU_net_utils.py', code_path)
    shutil.copy('./model/NYU_ViT.py', code_path)
    shutil.copy('./model/NYU_topk.py', code_path)
    writer1 = SummaryWriter('./result/NYU_PatchSize8/test1/train')

    history = pd.DataFrame()

    TrainLoader, ValLoader = get_data_loader(i_fold)

    model = ViT(num_classes = 2,
                dim = 112,
                depth = 3,
                heads = 2,
                mlp_dim = 16,
                dropout = 0,
                emb_dropout = 0).to(DEVICE)

    lossfunction = nn.CrossEntropyLoss(reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs, eta_min=lr_decay)

    logging.info(ifold)

    for epoch in range(train_epochs):

        cur_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_epoch(model, TrainLoader, DEVICE, optimizer, lossfunction)
        val_loss, val_acc, score_auc, f1, precision, recall, specificity = val_epoch(model, ValLoader, DEVICE, lossfunction)

        lr_scheduler.step()

        _h = pd.DataFrame(
            {'lr': [cur_lr], 'train_loss': [train_loss], 'train_acc': [train_acc], 'val_loss': [val_loss], 'val_acc': [val_acc]})
        history = history.append(_h, ignore_index=True)

        msg = f"Epoch{epoch}, lr:{cur_lr:.4f}, train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}, val_loss:{val_loss:.4f}, val_acc:{val_acc:.4f}"
        logging.info(msg)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_auc = score_auc
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_specificity = specificity
            cnt_wait = 0
            model_path = os.path.join(save_path, f"model_fold{i_fold}.pth")
            torch.save(model.state_dict(), model_path)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            break
    train_loss1 = history['train_loss'].dropna()
    for i in range(np.array(train_loss1).shape[0]):
        writer1.add_scalar(f'loss/ifold{i_fold}', np.array(train_loss1)[i], i)

    return train_loss, train_acc, val_loss, best_val_acc, best_val_auc, best_f1, best_precision, best_recall, best_specificity


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--seed', type=int, default=424, help='Random seed.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    open_log(f'result/NYU_PatchSize8/test1')
    train_acc_list, val_acc_list, val_auc_list = [], [], []
    f1 = []
    precision = []
    recall = []
    specificity = []

    for ifold in range(5):

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        train_loss, train_acc, val_loss, val_acc, val_auc, best_f1, best_precision, best_recall, best_specificity = train(ifold, DEVICE)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        val_auc_list.append(val_auc)
        f1.append(best_f1)
        precision.append(best_precision)
        recall.append(best_recall)
        specificity.append(best_specificity)

    logging.info(f'train_acc_list {train_acc_list}')
    logging.info(f'val_acc_list {val_acc_list}')
    logging.info(f'val_auc_list {val_auc_list}')
    logging.info(f'val_F1_list {f1}')
    logging.info(f'val_precision_list {precision}')
    logging.info(f'val_recall_list {recall}')
    logging.info(f'val_specificity_list {specificity}')
    logging.info(f'train_acc {np.array(train_acc_list).mean():.4f}±{np.array(train_acc_list).std():.4f}')
    logging.info(f'val_acc {np.array(val_acc_list).mean():.4f}±{np.array(val_acc_list).std():.4f}')
    logging.info(f'val_auc {np.array(val_auc_list).mean():.4f}±{np.array(val_auc_list).std():.4f}')
    logging.info(f'val_F1 {np.array(f1).mean():.4f}±{np.array(f1).std():.4f}')
    logging.info(f'val_precision {np.array(precision).mean():.4f}±{np.array(precision).std():.4f}')
    logging.info(f'val_recall {np.array(recall).mean():.4f}±{np.array(recall).std():.4f}')
    logging.info(f'val_specificity {np.array(specificity).mean():.4f}±{np.array(specificity).std():.4f}')