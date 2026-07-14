# file: final_check.py
from config import *
from data_utils import *
from model import NeuMF
from evaluate_model import final_test_report, demo_recommendation # Gọi hàm từ file của bạn
import torch

def run_test():
    # 1. Load data
    df, u_map, i_map, n_users, n_items = load_and_preprocess(DATA_DIR / "ratings.csv")
    user_item_set = build_user_item_set(df)
    train_df, test_df = split_leave_one_out(df)
    eval_data = build_eval_data(test_df, user_item_set, n_items)

    # 2. Load model
    model = NeuMF(n_users, n_items)
    model_path = OUTPUT_DIR / "neumf_final.pth"
    
    if not model_path.exists():
        print(f"❌ Không tìm thấy model tại {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(DEVICE)

    # 3. Chạy đánh giá "Tính mới" (Truyền thêm biến n_items vào cuối)
    final_test_report(model, eval_data, n_items)

    # 4. Chạy Demo thực tế cho User ID số 1 (thay bằng ID bất kỳ trong data của bạn)
    demo_recommendation(model, 1, u_map, i_map, user_item_set, n_items)

if __name__ == "__main__":
    run_test()