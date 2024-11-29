"""
Utility functions for LightGCN.
"""
import world
import torch
from torch import nn, optim
import numpy as np
from dataloader import BasicDataset
from sklearn.metrics import roc_auc_score
from time import time

class BPRLoss:
    def __init__(self, recmodel: nn.Module, config: dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss += reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


def UniformSample_similarity_based(dataset, neg_ratio=1):
    """
    Samples positive and negative items based on category similarity.
    """
    dataset: BasicDataset
    allPos = dataset.allPos
    similarity_matrix = dataset.similarity_matrix.values
    S = []
    for user in range(dataset.n_users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        for _ in range(neg_ratio):
            pos_item = np.random.choice(posForUser)
            while True:
                neg_item = np.random.randint(0, dataset.m_items)
                if neg_item not in posForUser:
                    if similarity_matrix[pos_item, neg_item] < dataset.threshold:
                        break
            S.append([user, pos_item, neg_item])
    return np.array(S)


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def minibatch(*tensors, **kwargs):
    """
    Splits data into batches.
    """
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])
    for i in range(0, len(tensors[0]), batch_size):
        yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    """
    Shuffles multiple arrays while preserving their order.
    """
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)
    result = tuple(x[shuffle_indices] for x in arrays)
    return result


def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def NDCGatK_r(test_data, r, k):
    pred_data = r[:, :k]
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = min(k, len(items))
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    return np.sum(ndcg)
