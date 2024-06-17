import logging
import math
import torch
from modules.HMKGR import HMKGR
from utils.util import *
from utils.evaluate import *
from sklearn.metrics import roc_auc_score, f1_score


def train(args, data):
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """load model"""
    logging.info("begin load model ...")
    model = HMKGR(args, data)
    logging.info("model parameters: " + get_total_parameters(model))
    model.to(device)
    logging.info(model)

    """set parameters"""
    Ks = eval(args.Ks)

    """prepare optimizer"""
    logging.info("begin prepare optimizer ...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    """training"""
    logging.info("start training ...")
    train_data = data.train_data_with_neg
    test_data = data.test_data_with_neg
    batch_num = math.ceil(len(train_data) / args.batch_size)

    for epoch in range(1, args.epoch + 1):
        torch.cuda.empty_cache()
        if epoch % 1 == 0:
            index = np.arange(len(train_data))
            np.random.shuffle(index)
            train_data = train_data[index]
        model.train()
        loss, s = 0, 0
        for i in range(batch_num):
            user_index, item_index, labels = get_inputs(train_data, i * args.batch_size, (i + 1) * args.batch_size,
                                                        device)
            batch_loss = model(user_index, item_index, labels, mode="train")

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
            s += args.batch_size

        if epoch % 1 == 0 or epoch == 1:
            model.eval()
            test_auc, test_acc, test_f1 = ctr_eval(model, test_data, args.batch_size)
            logging.info('Epoch %d  test auc: %.4f  acc: %.4f  f1: %.4f' % (epoch, test_auc, test_acc, test_f1))
            _, metrics_dict = evaluate(model, data, Ks, device)
            logging.info(
                'Epoch: {:d} Precision [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(
                    epoch,
                    metrics_dict[1]['precision'], metrics_dict[2]['precision'], metrics_dict[5]['precision'],
                    metrics_dict[10]['precision'],
                    metrics_dict[20]['precision'], metrics_dict[50]['precision'], metrics_dict[100]['precision']
                ))
            logging.info(
                'Epoch: {:d} Recall [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(
                    epoch,
                    metrics_dict[1]['recall'], metrics_dict[2]['recall'], metrics_dict[5]['recall'],
                    metrics_dict[10]['recall'],
                    metrics_dict[20]['recall'], metrics_dict[50]['recall'], metrics_dict[100]['recall']
                ))
            logging.info(
                'Epoch: {:d} NDCG [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]\n'.format(
                    epoch,
                    metrics_dict[1]['ndcg'], metrics_dict[2]['ndcg'], metrics_dict[5]['ndcg'], metrics_dict[10]['ndcg'],
                    metrics_dict[20]['ndcg'], metrics_dict[50]['ndcg'], metrics_dict[100]['ndcg']
                ))


def ctr_eval(model, data, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        users = data[start:start + batch_size, 0]
        items = data[start:start + batch_size, 1]
        with torch.no_grad():
            scores = model(users, items, mode="topK")
        scores = scores.cpu().numpy()
        labels = data[start:start + batch_size, 2]

        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        acc = np.mean(np.equal(scores, labels))

        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)

        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(f1_list))


def get_inputs(data, start, end, device):
    user_index = data[start:end, 0]
    item_index = data[start:end, 1]
    labels = data[start:end, 2]

    return torch.LongTensor(user_index).to(device), \
        torch.LongTensor(item_index).to(device), \
        torch.FloatTensor(labels).to(device)
