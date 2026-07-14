import time
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from data_utils import *
from model import NeuMF
from evaluate_model import evaluate

def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for users, items, labels in pbar:
        users, items, labels = users.to(DEVICE), items.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(users, items), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)

def main():
    # 1. Tiền xử lý dữ liệu
    df, u_map, i_map, n_users, n_items = load_and_preprocess(DATA_DIR / "ratings.csv")
    train_df, test_df = split_leave_one_out(df)
    user_item_set = build_user_item_set(df)
    
    train_loader = DataLoader(MovieLensDataset(train_df, n_items, user_item_set), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    eval_data = build_eval_data(test_df, user_item_set, n_items)

    # 2. Khởi tạo mô hình
    model = NeuMF(n_users, n_items).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCELoss()

    # --- ĐOẠN SỬA LẠI ĐỂ FIX LỖI KEYERROR ---
    model_path = OUTPUT_DIR / "neumf_final.pth"
    if model_path.exists():
        print("--- Tìm thấy model cũ, đang kiểm tra cấu trúc để nạp... ---")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Nếu file lưu dạng Dict (kiểu mới)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("✅ Đã nạp thành công từ Dictionary.")
        # Nếu file lưu dạng trọng số thuần túy (kiểu cũ của bạn)
        else:
            model.load_state_dict(checkpoint)
            print("✅ Đã nạp thành công từ trọng số thuần túy.")
    # ----------------------------------------

    # 3. Vòng lặp huấn luyện
    best_hr = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        hr, ndcg = evaluate(model, eval_data)
        
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | HR@{TOP_K}: {hr:.4f} | NDCG@{TOP_K}: {ndcg:.4f}")
        
        if hr > best_hr:
            best_hr = hr
            # Lưu lại model tốt nhất (Lưu theo kiểu mới để lần sau nạp ko lỗi nữa)
            save_dict = {
                "model_state_dict": model.state_dict(),
                "config": {"emb_dim": EMB_DIM, "mlp_layers": MLP_LAYERS}
            }
            torch.save(save_dict, model_path)
            
            with open(OUTPUT_DIR / "user_mapping.pkl", "wb") as f: pickle.dump(u_map, f)
            with open(OUTPUT_DIR / "item_mapping.pkl", "wb") as f: pickle.dump(i_map, f)
            print(f"--- Đã lưu model tốt nhất mới với HR: {hr:.4f} ---")

if __name__ == "__main__":
    main()