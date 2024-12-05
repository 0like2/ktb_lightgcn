"""
Procedure for training and testing LightGCN with metadata and similarity-based graph.
"""
import world
import numpy as np
import torch
import utils
from time import time
import multiprocessing

CORES = multiprocessing.cpu_count() // 2


def BPR_train(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    """
    Trains LightGCN using BPR loss with similarity-based graph.
    """
    print("[DEBUG] Starting BPR_train...")
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    # Sampling positive and negative examples
    start_time = time()
    S = utils.UniformSample_similarity_based(dataset)  # Uniform sampling without threshold filtering
    sample_time = time() - start_time

    if len(S) == 0 or S.ndim < 2:
        print("[ERROR] Sampling returned empty or invalid results. Check the sampling logic.")
        return

    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    # Move data to device
    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)

    # Shuffle data
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    average_loss = 0.0

    # Training in batches
    for batch_i, (batch_users, batch_pos, batch_neg) in enumerate(
        utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])
    ):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        average_loss += cri
        if world.tensorboard and w is not None:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)

    average_loss /= total_batch
    train_time = time() - start_time

    print(f"[BPR Train] Epoch {epoch}, Loss: {average_loss:.4f}, Sample Time: {sample_time:.2f}s, Train Time: {train_time:.2f}s")
    return f"loss{average_loss:.3f}"


def test_one_batch(X):
    """
    Evaluates one batch of predictions and computes recall, precision, and NDCG.
    """
    sorted_items = X[0].numpy()  # Predicted ranking
    groundTrue = X[1]  # Ground truth items
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []

    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))

    return {
        'recall': np.array(recall),
        'precision': np.array(pre),
        'ndcg': np.array(ndcg),
    }


def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    """
    Tests the model using recall, precision, and NDCG metrics.
    """
    u_batch_size = world.config['test_u_batch_size']
    testDict = dataset.testDict  # Ground truth user-item pairs
    Recmodel.eval()  # Set the model to evaluation mode

    max_K = max(world.topks)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}

    pool = None
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)

    with torch.no_grad():
        users = list(testDict.keys())  # List of test users
        total_batch = len(users) // u_batch_size + 1

        users_list, rating_list, groundTrue_list = [], [], []

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            batch_users = [int(u) for u in batch_users[0]]
            allPos = []
            for u in batch_users:
                user_category = dataset.creators.iloc[u]['channel_category']
                pos_items = np.where(dataset.similarity_matrix[user_category] > 0)[0]
                allPos.append(pos_items)

            groundTrue = [testDict[u] for u in batch_users]

            batch_users_gpu = torch.Tensor(batch_users).long().to(world.device)
            rating = Recmodel.getUsersRating(
                batch_users_gpu,
                creators_metadata_tensor=torch.tensor(dataset.creators[['channel_category']].values,
                                                      device=world.device),
                items_metadata_tensor=torch.tensor(dataset.items[['item_category']].values, device=world.device),
                similarity_matrix=torch.tensor(dataset.similarity_matrix, device=world.device)
            )

            # Exclude training positives from recommendation
            exclude_index, exclude_items = [], []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)  # Mask training items

            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        X = zip(rating_list, groundTrue_list)

        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = [test_one_batch(x) for x in X]

        scale = float(u_batch_size / len(users))

        for result in pre_results:
            results['recall'] += result['recall'] * scale
            results['precision'] += result['precision'] * scale
            results['ndcg'] += result['ndcg'] * scale

    if world.tensorboard and w is not None:
        w.add_scalars(f'Test/Recall@{world.topks}',
                      {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
        w.add_scalars(f'Test/Precision@{world.topks}',
                      {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
        w.add_scalars(f'Test/NDCG@{world.topks}',
                      {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)

    if multicore == 1:
        pool.close()

    print(f"Test Results: {results}")
    return results
