import torch
import numpy as np
from tqdm import tqdm
from config import DEVICE, TOP_K

def get_metrics(rank_list, pos_item):
    """Tính toán các chỉ số cốt lõi cho 1 User"""
    if pos_item in rank_list:
        # 1. Hit Ratio
        hr = 1.0
        # 2. NDCG (Độ hiệu quả xếp hạng có phạt vị trí bằng Log)
        pos = rank_list.index(pos_item)
        ndcg = np.log(2) / np.log(pos + 2)
        # 3. MRR (Mean Reciprocal Rank) - Vị trí xuất hiện của phim đúng
        mrr = 1.0 / (pos + 1)
        return hr, ndcg, mrr
    return 0, 0, 0

def evaluate(model, eval_data):
    """Hàm đánh giá nhanh được gọi trong lúc huấn luyện (Train)"""
    model.eval()
    hrs, ndcgs = [], []
    
    with torch.no_grad():
        for user, pos_item, neg_items in eval_data:
            item_tensor = torch.tensor([pos_item] + neg_items).to(DEVICE)
            user_tensor = torch.tensor([user] * len(item_tensor)).to(DEVICE)
            
            preds = model(user_tensor, item_tensor).squeeze()
            _, indices = torch.topk(preds, TOP_K)
            
            rank_list = item_tensor[indices].cpu().numpy().tolist()
            
            if pos_item in rank_list:
                hrs.append(1)
                pos = rank_list.index(pos_item)
                ndcgs.append(np.log(2) / np.log(pos + 2))
            else:
                hrs.append(0)
                ndcgs.append(0)
                
    return np.mean(hrs), np.mean(ndcgs)

def final_test_report(model, eval_data, total_unique_items):
    """
    Hàm đánh giá chuyên sâu sau khi huấn luyện.
    Đã loại bỏ Precision/Recall dư thừa và tích hợp TÍNH MỚI: Coverage & Gini Index.
    """
    model.eval()
    metrics = {'HR': [], 'NDCG': [], 'MRR': []}
    
    # Tập hợp lưu trữ tất cả các item ID từng được mô hình gợi ý ra (để tính Coverage)
    all_recommended_items = set()
    # Từ điển đếm tần suất xuất hiện của từng item trong các danh sách gợi ý (để tính Gini)
    item_counts = {}

    print(f"\n🚀 Đang chạy đánh giá chuyên sâu trên {len(eval_data)} users...")
    
    with torch.no_grad():
        for user, pos_item, neg_items in tqdm(eval_data):
            item_tensor = torch.tensor([pos_item] + neg_items).to(DEVICE)
            user_tensor = torch.tensor([user] * len(item_tensor)).to(DEVICE)
            
            preds = model(user_tensor, item_tensor).squeeze()
            _, indices = torch.topk(preds, TOP_K)
            rank_list = item_tensor[indices].cpu().numpy().tolist()
            
            # Cập nhật các chỉ số truyền thống
            hr, ndcg, mrr = get_metrics(rank_list, pos_item)
            metrics['HR'].append(hr)
            metrics['NDCG'].append(ndcg)
            metrics['MRR'].append(mrr)
            
            # Phục vụ tính toán TÍNH MỚI (Coverage & Gini)
            for item in rank_list:
                all_recommended_items.add(item)
                item_counts[item] = item_counts.get(item, 0) + 1

    # ---- BẮT ĐẦU TÍNH TOÁN CÁC ĐÓNG GÓP MỚI ----
    # 1. Catalog Coverage (%)
    catalog_coverage = len(all_recommended_items) / total_unique_items
    
    # 2. Gini Index (Độ bất bình đẳng trong phân phối gợi ý)
    n = total_unique_items
    counts = np.array([item_counts.get(i, 0) for i in range(total_unique_items)])
    sorted_counts = np.sort(counts)
    index = np.arange(1, n + 1)
    gini_index = (np.sum((2 * index - n - 1) * sorted_counts)) / (n * np.sum(sorted_counts))
    # --------------------------------------------

    print("\n" + "="*50)
    print(f"{'CHỈ SỐ ĐÁNH GIÁ THUẬT TOÁN':<30} | {'KẾT QUẢ':<15}")
    print("-" * 50)
    for m, values in metrics.items():
        print(f"{m + ' @' + str(TOP_K):<30} | {np.mean(values):.4f}")
    
    print("-" * 50)
    print(f"{'CHỈ SỐ ĐÁNH GIÁ DOANH NGHIỆP (TÍNH MỚI)':<30} |")
    print("-" * 50)
    print(f"{'Catalog Coverage @' + str(TOP_K):<30} | {catalog_coverage:.4f} ({catalog_coverage*100:.2f}%)")
    print(f"{'Gini Index @' + str(TOP_K):<30} | {gini_index:.4f}")
    print("="*50)


def demo_recommendation(model, user_id, user_mapping, item_mapping, user_item_set, n_items, top_n=10):
    """
    Chức năng nghiệp vụ: Tối ưu sạch sẽ, loại bỏ in terminal không cần thiết.
    Chỉ thực hiện biến đổi dải điểm dữ liệu thuần để trả về API chạy mượt mà trên Web.
    """
    model.eval()
    inv_item_map = {v: k for k, v in item_mapping.items()}
    u_idx = user_mapping.get(user_id)
    
    if u_idx is None:
        return []

    interacted = user_item_set.get(u_idx, set())
    candidates = [i for i in range(n_items) if i not in interacted]
    
    u_tensor = torch.tensor([u_idx] * len(candidates)).to(DEVICE)
    i_tensor = torch.tensor(candidates).to(DEVICE)
    
    with torch.no_grad():
        preds = model(u_tensor, i_tensor).squeeze()
        scores = preds.cpu().numpy()
        
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score > 1e-6:
            normalized_scores = (scores - min_score) / (max_score - min_score)
            beautiful_scores = normalized_scores * (0.965 - 0.05) + 0.05
        else:
            beautiful_scores = scores
        
        best_indices = np.argsort(beautiful_scores)[::-1][:top_n]
    
    recommendation_results = []
    for idx in best_indices:
        movie_idx = candidates[idx]
        actual_movie_id = int(inv_item_map[movie_idx])
        score_formatted = float(beautiful_scores[idx])
        
        recommendation_results.append({
            "movie_id": actual_movie_id,
            "score": score_formatted
        })
        
    return recommendation_results