import torch
import torch.nn as nn

class TMFG_Relation_Layer(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x_proj = self.projection(x)
        attn_out, _ = self.attn(x_proj, x_proj, x_proj)
        return self.norm(x_proj + self.dropout(attn_out))

class StockClassifierModel(nn.Module):
    def __init__(self, num_sectors=10, input_feat_dim=5, hidden_dim=128, n_classes=3):
        super().__init__()
        # 입력 차원 유연성 확보 (기본값 사용)
        self.trend_conv = nn.Conv1d(input_feat_dim, input_feat_dim, kernel_size=5, padding=2, groups=input_feat_dim)
        self.lstm = nn.LSTM(input_feat_dim * 2, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.bn_lstm = nn.BatchNorm1d(hidden_dim)
        self.inter_fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 7))
        self.relation_layer = TMFG_Relation_Layer(input_dim=7, embed_dim=128, num_heads=4)
        
        # 코스피 모델과 미국 모델의 출력 레이어 구조 통일 (기본적으로 이 구조를 따름)
        self.final_fc = nn.Sequential(
            nn.Linear(num_sectors * 128, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        
    def forward(self, x):
        b, s, t, f = x.shape
        x_flat = x.view(-1, t, f)
        x_perm = x_flat.permute(0, 2, 1)
        trend = self.trend_conv(x_perm).permute(0, 2, 1)
        # 잔차 연결 (Trend + Residual)
        combined = torch.cat([trend, x_flat - trend], dim=-1)
        
        _, (h, _) = self.lstm(combined)
        emb = self.bn_lstm(h[-1])
        stock_preds = self.inter_fc(emb).view(b, s, 7)
        enhanced = self.relation_layer(stock_preds)
        
        # 최종 예측
        return self.final_fc(enhanced.view(b, -1))
