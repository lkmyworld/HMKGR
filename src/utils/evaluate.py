import numpy as np
import torch


def topk_settings(train_data, test_data, n_item):
    user_num = 200
    k_list = [1, 2, 5, 10, 20, 50, 100]
    train_record = get_user_record(train_data)
    test_record = get_user_record(test_data)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))
    return user_list, train_record, test_record, item_set, k_list


def get_user_record(data):
    user_history_dict = dict()
    for user, item in zip(data[0], data[1]):
        if user not in user_history_dict:
            user_history_dict[user] = set()
        user_history_dict[user].add(item)
    return user_history_dict


def topk_eval(model, user_list, train_record, test_record, item_set, k_list, batch_size, device):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: 0 for m in metric_names} for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            users = [user] * batch_size
            items = test_item_list[start:start + batch_size]
            users = torch.LongTensor(users).to(device)
            items = torch.LongTensor(items).to(device)
            # labels = torch.FloatTensor(label).to(device)
            with torch.no_grad():
                scores = model(users, items, mode="topK")
            items = items.cpu().numpy()
            scores = scores.cpu().numpy()
            # scores = model(users, items, labels, mode='predict')
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            users = [user] * batch_size
            items = test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start)
            users = torch.LongTensor(users).to(device)
            items = torch.LongTensor(items).to(device)
            # labels = torch.FloatTensor(label).to(device)

            with torch.no_grad():
                scores = model(users, items, mode="topK")
            items = items.cpu().numpy()
            scores = scores.cpu().numpy()
            # scores = model(users, items, labels, mode='predict')
            for item, score in zip(items, scores):
                item_score_map[item] = score

        # top_k_items = heapq.nlargest(101, item_score_map.items(), key=lambda x: x[1])
        # item_sorted = [i[0] for i in top_k_items]
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        test_pos_item_binary = np.zeros(len(item_set), dtype=np.float32)
        test_pos_item_binary[list(test_record[user])] = 1
        binary_hit = test_pos_item_binary[item_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            ndcg = ndcg_at_k_batch1(binary_hit, k)

            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))
            ndcg_list[k].append(ndcg)

    for k in k_list:
        for m in metric_names:
            if m == 'precision':
                metrics_dict[k][m] = np.mean(precision_list[k])
            if m == 'recall':
                metrics_dict[k][m] = np.mean(recall_list[k])
            if m == 'ndcg':
                metrics_dict[k][m] = np.mean(ndcg_list[k])

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, ndcg, metrics_dict


def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    for batch_user_ids in user_ids_batches:
        batch_user_ids = batch_user_ids.to(device)

        with torch.no_grad():
            batch_scores = model(batch_user_ids, item_ids, mode='ctr')

        batch_scores = batch_scores.cpu()
        batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict,
                                          batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)

        cf_scores.append(batch_scores.numpy())
        for k in Ks:
            for m in metric_names:
                metrics_dict[k][m].append(batch_metrics[k][m])

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict


def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, Ks):
    """
    cf_scores: (n_users, n_items)
    """
    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
    for idx, u in enumerate(user_ids):
        train_pos_item_list = train_user_dict[u]
        test_pos_item_list = test_user_dict[u]
        cf_scores[idx][train_pos_item_list] = -np.inf
        test_pos_item_binary[idx][test_pos_item_list] = 1

    try:
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)  # try to speed up the sorting process
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(user_ids)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)

    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]['precision'] = precision_at_k_batch(binary_hit, k)
        metrics_dict[k]['recall'] = recall_at_k_batch(binary_hit, k)
        metrics_dict[k]['ndcg'] = ndcg_at_k_batch(binary_hit, k)
    return metrics_dict


def precision_at_k_batch(hits, k):
    """
    calculate Precision@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = hits[:, :k].mean(axis=1)
    return res


def ndcg_at_k_batch1(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits_k = hits[:k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)))

    sorted_hits_k = np.flip(np.sort(hits))[:k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)))

    if idcg == 0:
        idcg = np.inf
    # idcg[idcg == 0] = np.inf
    ndcg = (dcg / idcg)
    return ndcg


def ndcg_at_k_batch(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    idcg[idcg == 0] = np.inf
    ndcg = (dcg / idcg)
    return ndcg


def recall_at_k_batch(hits, k):
    """
    calculate Recall@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    return res
