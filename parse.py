import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--latent_dim', type=int, default=64,  # 이름 변경
                        help="the embedding size of lightGCN")
    parser.add_argument('--n_layers', type=int, default=3,  # 이름 변경
                        help="the number of layers in LightGCN")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalization")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="dropout keep probability")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="lgn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--multicore', type=int, default=0,
                        help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020,
                        help='random seed')
    parser.add_argument('--model', type=str, default='lgn',
                        help='rec-model, support [mf, lgn]')
    parser.add_argument('--top_k', type=int, default=3,
                        help="Number of top recommendations to generate.")
    parser.add_argument('--train_new_data', type=int, default=0,
                        help="Flag to indicate training or recommending on new data (0 for disable, 1 for enable).")
    return parser.parse_args()
