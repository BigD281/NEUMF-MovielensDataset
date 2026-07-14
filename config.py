import torch
from pathlib import Path
import random
import numpy as np

# Đường dẫn
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Siêu tham số
EMB_DIM = 32
MLP_LAYERS = [64, 32, 16, 8]
NEG_SAMPLES = 4
BATCH_SIZE = 4096  # Tối ưu cho GPU Colab
LR = 1e-3
NUM_WORKERS = 4   # Để 0 để tránh lỗi RAM trên Colab
NUM_EPOCHS = 20

# Cấu hình đánh giá (Dành cho HR@10, NDCG@10)
TOP_K = 10
EVAL_NEG = 99      # 99 mẫu âm + 1 mẫu dương = 100 mẫu để xếp hạng

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)