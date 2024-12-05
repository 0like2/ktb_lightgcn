import os
from os.path import join
import torch
import multiprocessing
from parse import parse_args

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

# Define paths
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
INPUT_PATH = join(ROOT_PATH, 'input')
OUTPUT_PATH = join(ROOT_PATH, 'output')
BOARD_PATH = join(OUTPUT_PATH, 'runs')
FILE_PATH = join(OUTPUT_PATH, 'checkpoints')


# Ensure necessary directories exist
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

# Configuration dictionary
config = {}
all_datasets = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'custom-similarity']
all_models = ['mf', 'lgn']

# Basic configurations from args
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim'] = args.latent_dim
config['n_layers'] = args.n_layers
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False  # Default setting
config['bigdata'] = False  # Default setting

# 상대 경로를 활용해 파일 경로 설정
config['creator_file'] = os.path.join(INPUT_PATH, 'Creator_random25.csv')
config['item_file'] = os.path.join(INPUT_PATH, 'Item_random25.csv')
config['similarity_matrix_file'] = os.path.join(INPUT_PATH, 'similarity_matrix.csv')

# Device configuration
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else 'cpu')
CORES = multiprocessing.cpu_count() // 2  # Use half the available cores
seed = args.seed  # Random seed for reproducibility

# Dataset and model settings
dataset = args.dataset
model_name = args.model

if dataset not in all_datasets:
    raise ValueError(f"Dataset '{dataset}' not supported! Available datasets: {all_datasets}")
if model_name not in all_models:
    raise ValueError(f"Model '{model_name}' not supported! Available models: {all_models}")

# Training and evaluation settings
TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)  # Convert string "[20]" to list [20]
tensorboard = args.tensorboard
comment = args.comment

# Suppress pandas warnings
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)


# Utility function for colored print
def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")


# Optional logo for display
logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# Uncomment to display the logo
# print(logo)

print("Configuration:")
for key, value in config.items():
    print(f"{key}: {value}")
