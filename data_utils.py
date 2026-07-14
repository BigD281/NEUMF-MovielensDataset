import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from config import NEG_SAMPLES, EVAL_NEG

def load_and_preprocess(ratings_path):
    print("▶ Đang đọc dữ liệu...")
    df = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"],
                     dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})
    
    unique_users, unique_items = df["userId"].unique(), df["movieId"].unique()
    user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
    item_mapping = {iid: idx for idx, iid in enumerate(unique_items)}
    
    df["user"] = df["userId"].map(user_mapping).astype("int32")
    df["item"] = df["movieId"].map(item_mapping).astype("int32")
    return df[["user", "item", "rating"]], user_mapping, item_mapping, len(unique_users), len(unique_items)

def split_leave_one_out(df):
    print("▶ Đang chia tập train / test (Leave-One-Out)...")
    # Lấy tương tác cuối cùng của mỗi user làm tập Test
    test_df = df.groupby("user", sort=False).tail(1).copy()
    train_df = df.drop(index=test_df.index).copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def build_user_item_set(df):
    return df.groupby("user")["item"].apply(set).to_dict()

class MovieLensDataset(Dataset):
    def __init__(self, df, num_items, user_item_set, num_neg=NEG_SAMPLES):
        self.num_items = num_items
        self.user_item_set = user_item_set
        self.num_neg = num_neg
        self.users = df["user"].values
        self.items = df["item"].values

    def __len__(self):
        return len(self.users) * (1 + self.num_neg)

    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.num_neg)
        sample_type = idx % (1 + self.num_neg)
        user = self.users[pos_idx]
        if sample_type == 0:
            return torch.tensor(user, dtype=torch.long), torch.tensor(self.items[pos_idx], dtype=torch.long), torch.tensor(1.0, dtype=torch.float32)
        else:
            interacted = self.user_item_set.get(int(user), set())
            while True:
                neg_item = random.randint(0, self.num_items - 1)
                if neg_item not in interacted: break
            return torch.tensor(user, dtype=torch.long), torch.tensor(neg_item, dtype=torch.long), torch.tensor(0.0, dtype=torch.float32)

def build_eval_data(test_df, user_item_set, num_items):
    print("▶ Đang xây dựng dữ liệu đánh giá (1 Pos + 99 Neg)...")
    eval_data = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Build Eval"):
        user, pos_item = int(row["user"]), int(row["item"])
        interacted = user_item_set.get(user, set())
        neg_items = []
        while len(neg_items) < EVAL_NEG:
            neg = random.randint(0, num_items - 1)
            if neg not in interacted and neg != pos_item:
                neg_items.append(neg)
        eval_data.append((user, pos_item, neg_items))
    return eval_data