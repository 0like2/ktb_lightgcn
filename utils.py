"""
Utility functions for LightGCN.
"""
import world
import torch
from torch import nn, optim
import numpy as np
from dataloader import BasicDataset
import os


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
        loss.backward(retain_graph=True)
        self.opt.step()

        print(f"[DEBUG] Total Loss (with Regularization): {loss.item()}")
        return loss.cpu().item()


def UniformSample_similarity_based(dataset):
    allPos = dataset.allPos  # 긍정적 아이템
    allNeg = dataset.allNeg  # 부정적 아이템
    n_users = dataset.n_users
    S = []

    for user in range(n_users):
        pos_items = allPos[user]
        neg_items = allNeg[user]

        if not neg_items:  # 부정적 아이템이 없으면 건너뜀
            print(f"[WARNING] User {user} has no negative items.")
            continue

        for pos_item in pos_items:
            neg_item = np.random.choice(neg_items)  # 부정적 아이템 랜덤 선택
            S.append([user, pos_item, neg_item])

    return np.array(S)



def getFileName():

    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-{world.config['latent_dim']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['n_layers']}-{world.config['latent_dim']}.pth.tar"
    else:
        raise ValueError(f"Unknown model name: {world.model_name}")
    return os.path.join(world.FILE_PATH, file)


def set_seed(seed):

    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])
    for i in range(0, len(tensors[0]), batch_size):
        yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)
    result = tuple(x[shuffle_indices] for x in arrays)
    return result


def RecallPrecision_ATk(test_data, r, k):
    # Test Data 디버깅
    if not test_data:
        print("[ERROR] Test data is empty. Ensure `testDict` is populated correctly.")
        return {'recall': np.nan, 'precision': 0}

    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])

    if np.any(recall_n == 0):
        print("[WARNING] Some users have no ground truth items in test data.")
        recall_n[recall_n == 0] = 1
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

def getLabel(test_data, pred_data):

    r = []
    for groundTrue, predictTopK in zip(test_data, pred_data):
        pred = np.array([1.0 if x in groundTrue else 0.0 for x in predictTopK])
        r.append(pred)
    return np.array(r)
