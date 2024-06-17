import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="HMKGR")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dataset', type=str, default='steam')
    # parser.add_argument('--dataset', type=str, default='movielens')

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--relation_dim', type=int, default=32)
    parser.add_argument("--cf_l2loss_lambda", type=float, default=1e-4)

    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-03)

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')

    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--test_batch_size', type=int, default=10000)
    parser.add_argument('--Ks', default='[1, 2, 5, 10, 20, 50, 100]')

    parser.add_argument("--node_dropout", type=bool, default=False, help="consider dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0, help="ratio of dropout")

    return parser.parse_args()
