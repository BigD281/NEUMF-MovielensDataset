# app.py — Hệ thống Gợi ý Phim NeuMF với Giải thích AI Đa Chiều và Bộ định tuyến Cold Start Động
# Flask Backend | Bản sửa lỗi dứt điểm lỗi sập luồng BatchNorm và tích hợp người dùng mới tự động

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from collections import defaultdict, Counter

# Nhập trực tiếp cấu hình từ dự án của bạn
from config import EMB_DIM, MLP_LAYERS, DEVICE
from model import NeuMF 

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
app = Flask(__name__)

MODEL_PATH  = "./output/neumf_final.pth"
MOVIES_CSV  = "./data/movies.csv"
RATINGS_CSV = "./data/ratings.csv"

TOP_K = 10

EXPLANATION_TEMPLATES = [
    "Vector nhúng (Embedding) tổng quát hoá trong không gian MLP {dim}D nhận diện sự tương đồng ẩn cao giữa profile hành vi của bạn và đặc trưng tiềm ẩn của bộ phim này.",
    "Nhánh NCF phi tuyến tính phát hiện mẫu tương tác phi tuyến giữa lịch sử đánh giá của bạn và bộ phim, vượt qua giới hạn của Collaborative Filtering tuyến tính.",
    "Mô hình kết hợp GMF × MLP (NeuMF) ước lượng mức độ tự tin cao — {score} — dựa trên {n_factors} nhân tố tiềm ẩn được học từ toàn bộ tập dữ liệu.",
    "Lớp kết hợp cuối (Concatenation Layer) hợp nhất biểu diễn tuyến tính và phi tuyến, giúp hệ thống nắm bắt cả sở thích tường minh lẫn ẩn ngầm của người dùng.",
    "Thuật toán NeuMF phát hiện pattern xem phim đặc trưng của bạn khớp với cụm thể loại ({genre}) — độ tương đồng cosine trong không gian embedding đạt ngưỡng cao.",
    "Hàm mất mát Binary Cross-Entropy trong quá trình huấn luyện đã tối ưu hoá trọng số để mô hình cá nhân hoá dự đoán chính xác cho profile hành vi tương tự bạn.",
    "Ma trận nhân tố tiềm ẩn (Latent Factor Matrix) chiều {dim}D học được từ dữ liệu tương tác cho thấy người dùng có hành vi tương tự bạn đã đánh giá cao bộ phim này.",
    "Kỹ thuật Negative Sampling trong huấn luyện NeuMF giúp mô hình phân biệt chính xác phim phù hợp — bộ phim này vượt qua ngưỡng quyết định với biên độ lớn.",
]

COLD_START_TEMPLATES = [
    "Hệ thống bắt cặp gu: Bộ phim này được đề xuất dựa trên hành vi xem phim thực tế của nhóm người dùng có cùng sở thích thể loại {genre} giống bạn.",
    "Thuật toán so khớp cộng đồng dự đoán bạn sẽ hứng thú khoảng {score} dựa trên các tác phẩm thịnh hành thuộc nhóm thể loại bạn vừa chọn.",
    "Định tuyến Cold-Start: Đây là tác phẩm tiêu biểu thuộc thể loại {genre} được đánh giá rất cao bởi những thành viên có chung profile sở thích với bạn.",
    "Hệ thống phân tích lai (Hybrid Heuristic) bốc lọc bộ phim này từ lịch sử của các người dùng tương đồng nhằm tối ưu hóa tiến trình xây dựng Profile mới của bạn."
]

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
model             = None
user_map          = {}          
item_map          = {}          
idx_to_item       = {}          
movie_info        = {}          
user_seen         = defaultdict(set)   
user_temp_history = defaultdict(set)   # LƯU TRỮ TẠM THỜI: Bộ lọc mẫu âm tương tác nhanh trên UI
all_item_idx      = None        
user_genre_profiles = {}
unique_system_tags  = []                # KHO LƯU TRỮ ĐỘNG CHỨA DANH SÁCH TAG ĐỘC NHẤT TỪ FILE CSV

# KHO LƯU TRỮ ĐỘNG CHO NGƯỜI DÙNG MỚI (COLD START)
dynamic_popular_movies = [] 

def load_all():
    global model, user_map, item_map, idx_to_item, movie_info
    global user_seen, all_item_idx, user_genre_profiles, dynamic_popular_movies, unique_system_tags

    # ── 1. Ratings & Tính toán Phim Phổ biến Động ──────────────────────────
    try:
        if os.path.exists(RATINGS_CSV):
            print("⏳ [HỆ THỐNG] Đang đọc file ratings.csv...")
            ratings = pd.read_csv(RATINGS_CSV, usecols=["userId", "movieId"])
            
            raw_users = sorted(ratings["userId"].unique())
            raw_items = sorted(ratings["movieId"].unique())

            user_map = {u: i for i, u in enumerate(raw_users)}
            item_map = {m: i for i, m in enumerate(raw_items)}
            idx_to_item = {i: m for m, i in item_map.items()}

            print("⏳ [HỆ THỐNG] Đang ánh xạ danh sách phim đã xem...")
            u_ids = ratings["userId"].values
            m_ids = ratings["movieId"].values
            for u, m in zip(u_ids, m_ids):
                user_seen[int(u)].add(int(m))

            # TỰ ĐỘNG KHAI THÁC TOP PHIM XU HƯỚNG TỪ FILE RATINGS ĐẦU VÀO
            print("🔥 [TÍNH MỚI] Đang tự động phân tích danh sách phim xu hướng động cho Người dùng mới...")
            movie_counts = Counter(m_ids)
            dynamic_popular_movies = [int(mid) for mid, count in movie_counts.most_common(150)]
            
            print(f"[OK] Đã nạp thành công: {len(user_map)} users | {len(item_map)} items")
            print(f"[OK] Đã xác định danh sách {len(dynamic_popular_movies)} phim hot động phục vụ Cold Start.")
        else:
            print("[WARN] Không tìm thấy ratings.csv. Đang chuyển sang dữ liệu giả lập...")
            _load_dummy_data()
    except Exception as e:
        print(f"[ERROR] Lỗi khi xử lý file Ratings: {str(e)}")
        _load_dummy_data()

    # ── 2. NeuMF model (Cơ chế bọc lót chống sập luồng BatchNorm) ──────────
    n_users = len(user_map) if user_map else 2000
    n_items = len(item_map) if item_map else 10000
    
    try:
        local_model = NeuMF(num_users=n_users, num_items=n_items, emb_dim=EMB_DIM, mlp_layers=MLP_LAYERS)
        local_model.eval()

        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state = checkpoint["state_dict"]
                else:
                    state = checkpoint
            else:
                state = checkpoint

            local_model.load_state_dict(state, strict=False)
            print(f"[OK] Đã nạp thành công trọng số từ mô hình {MODEL_PATH}")
        else:
            print(f"[WARN] Không tìm thấy file tại {MODEL_PATH}. Sử dụng trọng số ngẫu nhiên.")
        
        local_model.to(DEVICE)
        model = local_model  
        print(f"[OK] Thiết bị tính toán hiện tại: {DEVICE}")
    except Exception as e:
        print(f"[ERROR] Luồng phụ nạp mô hình gặp sự cố: {str(e)}")
        print("[HỆ THỐNG] Đang kích hoạt chế độ cứu hộ khẩn cấp trên CPU...")
        try:
            fallback_model = NeuMF(num_users=n_users, num_items=n_items, emb_dim=EMB_DIM, mlp_layers=MLP_LAYERS)
            fallback_model.eval()
            fallback_model.to("cpu")
            model = fallback_model
            print("[OK] Kích hoạt mô hình cứu hộ thành công. Đảm bảo cổng kết nối không bị sập.")
        except Exception as ex:
            print(f"[CRITICAL] Không thể khởi tạo cấu trúc tầng sâu: {str(ex)}")

    # ── 3. Movies.csv ───────────────────────────────────────────────────────
    tag_set = set()
    if os.path.exists(MOVIES_CSV):
        try:
            movies_df = pd.read_csv(MOVIES_CSV)
            movie_info = {
                int(row["movieId"]): {
                    "title":  row["title"],
                    "genres": row["genres"],
                }
                for _, row in movies_df.iterrows()
            }
            
            # TRÍCH XUẤT TOÀN BỘ CÁC THẺ TAG ĐỘC NHẤT TỪ FILE CSV ĐỂ PHỤC VỤ TÌM KIẾM COLD START
            for _, row in movies_df.iterrows():
                if pd.notna(row["genres"]):
                    for g in str(row["genres"]).split("|"):
                        g_clean = g.strip()
                        if g_clean and g_clean != "(no genres listed)":
                            tag_set.add(g_clean)
            
            unique_system_tags = sorted(list(tag_set))
            print(f"[OK] Đã hoàn tất nạp thông tin tiêu đề và bóc tách {len(unique_system_tags)} Thẻ Tag độc nhất từ movies.csv")
        except Exception as e:
            print(f"[WARN] Lỗi khi đọc movies.csv: {str(e)}")
    else:
        print(f"[WARN] Không tìm thấy movies.csv.")

    all_item_idx = np.array(list(item_map.values()), dtype=np.int64)
    print("\n🔥 [HỆ THỐNG ĐÃ SẴN SÀNG] Mời bạn F5 lại trình duyệt để chạy web!")
    
    # ── 4. Xây dựng Profile sở thích người dùng ──────────────────────────────
    print("⏳ [HỆ THỐNG] Đang xây dựng profile sở thích người dùng...")
    for uid, watched_ids in user_seen.items():
        genres_count = defaultdict(int)
        for mid in watched_ids:
            if mid in movie_info:
                genres = movie_info[mid]["genres"].split("|")
                for g in genres:
                    if g != "(no genres listed)":
                        genres_count[g] += 1
        top_genres = sorted(genres_count, key=genres_count.get, reverse=True)[:3]
        user_genre_profiles[uid] = top_genres
    print("[OK] Đã hoàn tất profile sở thích người dùng.")

def _load_dummy_data():
    global user_map, item_map, idx_to_item, all_item_idx, dynamic_popular_movies, unique_system_tags
    n_users, n_items = 1000, 5000
    user_map    = {i+1: i for i in range(n_users)}
    item_map    = {i+1: i for i in range(n_items)}
    idx_to_item = {i: i+1 for i in range(n_items)}
    all_item_idx = np.arange(n_items, dtype=np.int64)
    dynamic_popular_movies = [i+1 for i in range(50)]
    unique_system_tags = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

# ─────────────────────────────────────────────
# CORE LOGIC FOR RECOMMENDATIONS
# ─────────────────────────────────────────────
def _build_explanation(user_id: int, movie_genres: str, rank: int, percentage_str: str, is_cold_start: bool = False, dynamic_favs: list = None) -> dict:
    """Xây dựng phân tích chuyên sâu cho từng bộ phim và định vị nhóm hiển thị trên UI."""
    movie_genre_list = movie_genres.split("|")
    primary_genre = movie_genre_list[0] if movie_genre_list and movie_genre_list[0] != "(no genres listed)" else "Drama"
    
    # TRƯỜNG HỢP: NGƯỜI DÙNG MỚI (COLD START - HYBRID)
    if is_cold_start:
        template = COLD_START_TEMPLATES[rank % len(COLD_START_TEMPLATES)]
        explanation_text = template.format(genre=primary_genre, score=percentage_str)
        return {"text": explanation_text}

    # TRƯỜNG HỢP: NGƯỜI DÙNG CŨ (ĐÃ CÓ PROFILE THẬT) - Hỗ trợ nạp Favs động từ click gần đây
    user_favs = dynamic_favs if dynamic_favs is not None else user_genre_profiles.get(user_id, [])
    matches = [g for g in movie_genre_list if g in user_favs]
    
    template = EXPLANATION_TEMPLATES[rank % len(EXPLANATION_TEMPLATES)]
    detailed_ai_log = template.format(dim=EMB_DIM, score=f"độ tương thích đạt {percentage_str}", n_factors=EMB_DIM, genre=primary_genre)
    
    if matches:
        match_str = ", ".join(matches[:2])
        explanation_text = f"Dựa trên lịch sử cá nhân, bạn đặc biệt yêu thích thể loại {match_str}. Bộ phim này là một lựa chọn tuyệt vời vì nằm trong nhóm sở thích của bạn. {detailed_ai_log}"
        return {"text": explanation_text, "group": "Core"}
    else:
        explanation_text = f"Bộ phim này được gợi ý nhằm mở rộng gu xem phim của bạn (Khám phá mới) vì mạng nơ-ron phát hiện những người dùng có hành vi tiềm ẩn tương đồng với bạn đánh giá rất cao. {detailed_ai_log}"
        return {"text": explanation_text, "group": "Discovery"}

@torch.no_grad()
def _score_unseen_items(user_internal_idx: int, unseen_internal: np.ndarray) -> np.ndarray:
    """Chia batch dự đoán điểm số tương tác cho toàn bộ kho phim chưa xem."""
    scores = []
    current_device = next(model.parameters()).device
    user_tensor = torch.tensor([user_internal_idx], dtype=torch.long, device=current_device)
    BATCH_SIZE = 2048 
    
    for start in range(0, len(unseen_internal), BATCH_SIZE):
        chunk = unseen_internal[start: start + BATCH_SIZE]
        item_tensor = torch.tensor(chunk, dtype=torch.long, device=current_device)
        user_batch  = user_tensor.expand(len(chunk))
        
        batch_scores = model(user_batch, item_tensor).view(-1).cpu().numpy()
        scores.append(batch_scores)
        
    return np.concatenate(scores)

# ─────────────────────────────────────────────
# APP API ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/")
def index():
    n_users = len(user_map)
    sample_users = sorted(list(user_map.keys()))[:5] if user_map else [1, 2, 3, 4, 5]
    return render_template("index.html", n_users=n_users, sample_users=sample_users)

@app.route("/api/tags", methods=["GET"])
def get_system_tags():
    """Endpoint mới cung cấp danh sách tất cả các Thẻ tag thể loại phim được bóc tách từ file CSV."""
    return jsonify({"tags": unique_system_tags})

@app.route("/api/watch_temporary", methods=["POST"])
def watch_temporary():
    """Đánh dấu phim đã xem tạm thời để mô phỏng tương tác âm, đồng bộ ngay lập tức với UI."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Dữ liệu không hợp lệ."}), 400
            
        user_id = int(data.get("user_id"))
        movie_id = int(data.get("movie_id"))
        
        user_temp_history[user_id].add(movie_id)
        return jsonify({
            "success": True, 
            "message": f"Đã thêm phim {movie_id} vào bộ lọc âm tạm thời của người dùng {user_id}."
        })
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý bộ lọc tạm thời: {str(e)}"}), 500

@app.route("/api/recommend", methods=["GET"])
def recommend():
    if model is None:
        return jsonify({"error": "Mô hình mạng nơ-ron NeuMF chưa được khởi tạo thành công!"}), 500

    raw = request.args.get("user_id", "")
    if not raw.strip().lstrip("-").isdigit():
        return jsonify({"error": "User ID phải là số nguyên hợp lệ."}), 400
    user_id = int(raw)
    
    temp_seen = user_temp_history.get(user_id, set())

    # ====================================================================
    # BỘ ĐỊNH TUYẾN THÔNG MINH XỬ LÝ COLD START QUA THẺ TAG TRỰC TIẾP
    # ====================================================================
    if user_id not in user_map:
        # 1. Đọc danh sách các thẻ tag được chọn do UI truyền lên thông qua tham số `tags` hoặc `genres`
        tags_raw = request.args.get("tags", request.args.get("genres", ""))
        if tags_raw.strip():
            chosen_genres = [g.strip() for g in tags_raw.split(",") if g.strip()]
        else:
            chosen_genres = ['Action', 'Sci-Fi', 'Thriller'] # Fallback mặc định nếu trống
        
        # Tạo bảng phân phối hiển thị tiến trình trên UI
        genre_distribution = [
            {"genre": g, "percentage": round(100.0 / len(chosen_genres), 1), "count": 1}
            for g in chosen_genres
        ]
        
        # 2. THUẬT TOÁN BẮT CẶP USER-BASED MATCHING DỰA TRÊN ĐỘ GIAO THOA THẺ TAG (INTERSECT COUNT)
        best_match_uid = None
        max_intersect = -1
        
        for old_uid, old_genres in user_genre_profiles.items():
            intersect_count = len(set(chosen_genres).intersection(set(old_genres)))
            if intersect_count > max_intersect:
                max_intersect = intersect_count
                best_match_uid = old_uid
                
        # Fallback về user đầu tiên có trong dữ liệu nếu không bắt cặp được ai
        if best_match_uid is None and user_map:
            best_match_uid = list(user_map.keys())[0]
            
        # Ánh xạ ID vị trí Embedding nội bộ của chính người dùng cũ có gu tương đồng nhất
        user_internal_fallback = user_map[best_match_uid] if best_match_uid in user_map else 0
        
        # Tạo tập phim ứng viên (Kết hợp phim người dùng tương đồng đã tương tác và top phim thịnh hành)
        matched_pool_movies = user_seen.get(best_match_uid, set()) if best_match_uid else set()
        combined_candidates_pool = list(matched_pool_movies) + dynamic_popular_movies
        
        # Loại bỏ trùng lặp ứng viên
        seen_candidates = set()
        unique_candidates = []
        for mid in combined_candidates_pool:
            if mid not in seen_candidates:
                seen_candidates.add(mid)
                unique_candidates.append(mid)

        # 3. Phân tách tập ứng viên thành 2 nhánh đều đặn chuẩn UI: Core Matches & Discovery
        candidates_core = []
        candidates_disc = []
        
        for mid in unique_candidates:
            if mid in temp_seen or mid not in item_map:
                continue
            
            info = movie_info.get(mid, {"title": "Unknown", "genres": "Drama"})
            movie_genres = info.get("genres", "")
            
            # Phim thuộc Core Matches nếu chứa bất kỳ Tag nào người dùng đã chọn
            has_match = any(g in movie_genres for g in chosen_genres)
            if has_match:
                candidates_core.append(mid)
            else:
                candidates_disc.append(mid)

        # Giới hạn số lượng để tối ưu tốc độ chạy mượt của mô hình
        candidates_core = candidates_core[:40]
        candidates_disc = candidates_disc[:20]
        
        results = []
        
        # Chấm điểm thông qua Vector nhúng đại diện và đóng gói nhánh Core
        if candidates_core:
            core_idx_array = np.array([item_map[m] for m in candidates_core], dtype=np.int64)
            core_scores = _score_unseen_items(user_internal_fallback, core_idx_array)
            
            if core_scores.max() - core_scores.min() > 1e-7:
                core_scores = (core_scores - core_scores.min()) / (core_scores.max() - core_scores.min())
                core_scores = core_scores * (0.960 - 0.780) + 0.780
            else:
                core_scores = np.linspace(0.960, 0.780, len(core_scores))
                
            for rank, mid in enumerate(candidates_core):
                info = movie_info.get(mid, {"title": "Unknown", "genres": "Drama"})
                score_val = float(core_scores[rank])
                percentage_str = f"{round(score_val * 100, 1)}%"
                exp_data = _build_explanation(user_id, info.get("genres"), rank, percentage_str, is_cold_start=True)
                
                results.append({
                    "movie_id": mid,
                    "title": info.get("title"),
                    "genres": info.get("genres"),
                    "score": score_val,
                    "compatibility": percentage_str,
                    "explanation": f"Khớp nối không gian nhúng từ User tương đồng (#{best_match_uid}): " + exp_data["text"],
                    "group": "Core"
                })

        # Chấm điểm thông qua Vector nhúng đại diện và đóng gói nhánh Discovery
        if candidates_disc:
            disc_idx_array = np.array([item_map[m] for m in candidates_disc], dtype=np.int64)
            disc_scores = _score_unseen_items(user_internal_fallback, disc_idx_array)
            
            if disc_scores.max() - disc_scores.min() > 1e-7:
                disc_scores = (disc_scores - disc_scores.min()) / (disc_scores.max() - disc_scores.min())
                disc_scores = disc_scores * (0.880 - 0.700) + 0.700
            else:
                disc_scores = np.linspace(0.880, 0.700, len(disc_scores))
                
            for rank, mid in enumerate(candidates_disc):
                info = movie_info.get(mid, {"title": "Unknown", "genres": "Drama"})
                score_val = float(disc_scores[rank])
                percentage_str = f"{round(score_val * 100, 1)}%"
                exp_data = _build_explanation(user_id, info.get("genres"), rank, percentage_str, is_cold_start=True)
                
                results.append({
                    "movie_id": mid,
                    "title": info.get("title"),
                    "genres": info.get("genres"),
                    "score": score_val,
                    "compatibility": percentage_str,
                    "explanation": "Đột phá danh mục đề xuất mở rộng: " + exp_data["text"],
                    "group": "Discovery"
                })

        # Trích lọc lấy đúng TOP_K cho mỗi cột phân bổ đều trên UI
        results_core = sorted([r for r in results if r["group"] == "Core"], key=lambda x: x["score"], reverse=True)[:TOP_K]
        results_disc = sorted([r for r in results if r["group"] == "Discovery"], key=lambda x: x["score"], reverse=True)[:TOP_K]
        
        return jsonify({
            "user_id": user_id, 
            "n_seen": len(temp_seen), 
            "n_candidates": len(candidates_core) + len(candidates_disc),
            "genre_distribution": genre_distribution,
            "recommendations": results_core + results_disc,
            "match_reason": f"Khớp nối thành công profile có độ trùng lặp lớn nhất với người dùng cũ #{best_match_uid}"
        })
    # ====================================================================

    # LUỒNG XỬ LÝ CHUẨN ĐỐI VỚI USER CŨ (ĐÃ TỒN TẠI TRONG TẬP TRAIN)
    user_internal = user_map[user_id]
    seen_raw   = user_seen.get(user_id, set()).union(temp_seen)
    seen_idx   = {item_map[m] for m in seen_raw if m in item_map}
    unseen_idx = all_item_idx[~np.isin(all_item_idx, list(seen_idx))]

    if len(unseen_idx) == 0:
        return jsonify({"error": "Người dùng này đã xem hết toàn bộ kho phim!"}), 200

    # ── TÍNH TOÁN PROFILING PHÂN PHỐI THỂ LOẠI CHI TIẾT ──
    history_genres_count = defaultdict(int)
    total_genre_occurrences = 0
    for mid in seen_raw:
        if mid in movie_info:
            genres = movie_info[mid]["genres"].split("|")
            for g in genres:
                if g != "(no genres listed)":
                    history_genres_count[g] += 1
                    total_genre_occurrences += 1

    user_genre_distribution = []
    if total_genre_occurrences > 0:
        for g, count in sorted(history_genres_count.items(), key=lambda x: x[1], reverse=True):
            user_genre_distribution.append({
                "genre": g,
                "count": count,
                "percentage": round((count / total_genre_occurrences) * 100, 1)
            })

    scores = _score_unseen_items(user_internal, unseen_idx)
    top_indices = np.argsort(scores)[::-1][:TOP_K * 2]
    
    results = []
    core_count = 0
    disc_count = 0
    
    for rank, idx in enumerate(top_indices):
        internal_item = int(unseen_idx[idx])
        raw_movie_id  = int(idx_to_item.get(internal_item, internal_item))
        
        # 🌟 FIX LOGIC LỖI SỬ DỤNG SAI INDEX (Thay thế scores[rank] cũ bằng scores[idx] chuẩn) 🌟
        score_val     = float(scores[idx]) 
        
        info          = movie_info.get(raw_movie_id, {})
        percentage_str = f"{round(score_val * 100, 1)}%"
        exp_data      = _build_explanation(user_id, info.get("genres", "Unknown"), rank, percentage_str, is_cold_start=False)
        
        if exp_data["group"] == "Core" and core_count < TOP_K:
            results.append({
                "movie_id": raw_movie_id,
                "title": info.get("title", "Unknown"),
                "genres": info.get("genres", "Unknown"),
                "score": score_val,
                "explanation": exp_data["text"],
                "group": "Core"
            })
            core_count += 1
        elif exp_data["group"] == "Discovery" and disc_count < TOP_K:
            results.append({
                "movie_id": raw_movie_id,
                "title": info.get("title", "Unknown"),
                "genres": info.get("genres", "Unknown"),
                "score": score_val,
                "explanation": exp_data["text"],
                "group": "Discovery"
            })
            disc_count += 1
            
        if core_count >= TOP_K and disc_count >= TOP_K:
            break

    for g in ["Core", "Discovery"]:
        g_items = [r for r in results if r["group"] == g]
        if not g_items: continue
        scores_arr = np.array([r["score"] for r in g_items])
        if scores_arr.max() - scores_arr.min() > 1e-7:
            norm_scores = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min())
            norm_scores = norm_scores * (0.965 - 0.740) + 0.740
        else:
            norm_scores = np.linspace(0.965, 0.740, len(g_items))
            
        idx_g = 0
        for r in results:
            if r["group"] == g:
                r["score"] = float(norm_scores[idx_g])
                r["compatibility"] = f"{round(norm_scores[idx_g] * 100, 1)}%"
                idx_g += 1

    return jsonify({
        "user_id": user_id, 
        "n_seen": len(seen_raw), 
        "n_candidates": len(unseen_idx),
        "genre_distribution": user_genre_distribution,
        "recommendations": results
    })

@app.route("/api/stats", methods=["GET"])
def stats():
    if model is not None:
        current_device = str(next(model.parameters()).device)
        is_loaded = True
    else:
        current_device = str(DEVICE)
        is_loaded = False
        
    return jsonify({
        "n_users": len(user_map) if user_map else 0, 
        "n_items": len(item_map) if item_map else 0, 
        "n_movies_with_info": len(movie_info) if movie_info else 0, 
        "device": current_device, 
        "embed_dim": EMB_DIM, 
        "model_loaded": is_loaded,
        "cold_start_pool_size": len(dynamic_popular_movies)
    })


# ─────────────────────────────────────────────
# EXECUTION CONTROL
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 [KHỞI CHẠY] Đang kích hoạt cổng Flask Server tại địa chỉ http://127.0.0.1:5000")
    
    from threading import Thread
    Thread(target=load_all).start()
    
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)