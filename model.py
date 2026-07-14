import torch
import torch.nn as nn
from config import EMB_DIM, MLP_LAYERS, DEVICE

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=EMB_DIM, mlp_layers=None, dropout=0.2):
        super().__init__()
        # Lấy mlp_layers từ config nếu không truyền vào
        self.mlp_layers = mlp_layers if mlp_layers is not None else MLP_LAYERS

        # --- GMF embeddings ---
        self.emb_user_gmf = nn.Embedding(num_users, emb_dim)
        self.emb_item_gmf = nn.Embedding(num_items, emb_dim)

        # --- MLP embeddings ---
        self.emb_user_mlp = nn.Embedding(num_users, emb_dim)
        self.emb_item_mlp = nn.Embedding(num_items, emb_dim)

        # --- MLP Layers ---
        mlp_modules = []
        in_size = emb_dim * 2
        for out_size in self.mlp_layers:
            mlp_modules.append(nn.Linear(in_size, out_size))
            mlp_modules.append(nn.BatchNorm1d(out_size)) # Giúp ổn định GPU
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout))
            in_size = out_size
        self.mlp = nn.Sequential(*mlp_modules)

        # --- Final Prediction Layer ---
        predict_in = emb_dim + self.mlp_layers[-1]
        self.predict_layer = nn.Linear(predict_in, 1)
        self.sigmoid = nn.Sigmoid()
        
        self._init_weights()
        
        # BƯỚC QUAN TRỌNG: Đẩy toàn bộ trọng số lên GPU ngay khi khởi tạo
        self.to(DEVICE)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, user, item):
        # Ép kiểu Long để tránh lỗi tensor trên GPU
        user = user.long()
        item = item.long()

        # GMF Part
        u_gmf = self.emb_user_gmf(user)
        i_gmf = self.emb_item_gmf(item)
        gmf_out = u_gmf * i_gmf

        # MLP Part
        u_mlp = self.emb_user_mlp(user)
        i_mlp = self.emb_item_mlp(item)
        mlp_in = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Kết hợp
        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        
        # Output dạng (batch_size,) thay vì (batch_size, 1)
        return self.sigmoid(self.predict_layer(concat)).view(-1)