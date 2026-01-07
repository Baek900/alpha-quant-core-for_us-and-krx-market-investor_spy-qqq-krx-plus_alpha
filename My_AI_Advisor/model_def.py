import torch
import torch.nn as nn
import torch.nn.functional as F

# [v2.0] Focal Loss 정의 (QQQ 및 KOSPI 모델에서 사용)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [Batch, N_Classes] (Logits), targets: [Batch] (Indices)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 모델이 정답을 맞출 확률
        
        # Focal Loss 수식: alpha * (1 - pt)^gamma * CE
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# [v2.0] TMFG 관계형 레이어 (주식 간의 관계 어텐션 분석)
class TMFG_Relation_Layer(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2 if embed_dim == 128 else 0.3) # 한국 모델은 0.3
        
    def forward(self, x):
        x_proj = self.projection(x)
        attn_out, _ = self.attn(x_proj, x_proj, x_proj)
        return self.norm(x_proj + self.dropout(attn_out))

# [v2.0] 통합 주식 분류 모델 (미국/한국 설정 대응)
class StockClassifierModel(nn.Module):
    def __init__(self, num_sectors=10, input_feat_dim=13, hidden_dim=128, n_classes=5, is_kr=False):
        super().__init__()
        # 1. 입력 정규화 및 트렌드 추출 (1D Conv)
        self.input_bn = nn.BatchNorm1d(input_feat_dim)
        self.trend_conv = nn.Conv1d(input_feat_dim, input_feat_dim, kernel_size=5, padding=2, groups=input_feat_dim)
        
        # 2. LSTM 엔진 (한국 모델은 Bidirectional=True, Dropout=0.3 적용)
        self.bidirectional = is_kr
        self.lstm = nn.LSTM(
            input_feat_dim * 2, 
            hidden_dim, 
            batch_first=True, 
            num_layers=2, 
            dropout=0.3 if is_kr else 0.2, 
            bidirectional=self.bidirectional
        )
        
        # 3. LSTM 출력 처리 (한국 모델은 LayerNorm 사용)
        lstm_out_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        if is_kr:
            self.ln_lstm = nn.LayerNorm(lstm_out_dim)
        else:
            self.bn_lstm = nn.BatchNorm1d(hidden_dim)
            
        # 4. 개별 종목 특징 추출 (Inter FC)
        inter_out_dim = 128 if is_kr else 64
        self.inter_fc = nn.Sequential(
            nn.Linear(lstm_out_dim, inter_out_dim),
            nn.GELU() if is_kr else nn.ReLU(),
            nn.Linear(inter_out_dim, 7)
        )
        
        # 5. 종목 간 관계 분석 (Relation Layer)
        self.relation_layer = TMFG_Relation_Layer(
            input_dim=7, 
            embed_dim=256 if is_kr else 128, 
            num_heads=4
        )
        
        # 6. 최종 분류 레이어 (한국 모델은 512-128-5 구조 및 LayerNorm/GELU 사용)
        if is_kr:
            self.final_fc = nn.Sequential(
                nn.Linear(num_sectors * 256, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Linear(128, n_classes)
            )
        else:
            self.final_fc = nn.Sequential(
                nn.Linear(num_sectors * 128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes)
            )
        
    def forward(self, x):
        # x: [Batch, Sectors, Time, Feat]
        b, s, t, f = x.shape
        x_flat = x.view(-1, t, f)
        
        # Input BN & Trend Conv
        x_perm = x_flat.permute(0, 2, 1)
        x_norm = self.input_bn(x_perm)
        trend = self.trend_conv(x_norm)
        
        # Residual Connection (Trend vs Variation)
        x_norm_t = x_norm.permute(0, 2, 1)
        trend_t = trend.permute(0, 2, 1)
        combined = torch.cat([trend_t, x_norm_t - trend_t], dim=-1)
        
        # LSTM Step
        lstm_out, (h, _) = self.lstm(combined)
        
        if self.bidirectional:
            # 한국 모델: 시퀀스 마지막 출력에 LayerNorm 적용
            emb = self.ln_lstm(lstm_out[:, -1, :])
        else:
            # 미국 모델: 마지막 레이어의 마지막 Hidden State에 BatchNorm 적용
            emb = self.bn_lstm(h[-1])
            
        # Per-Stock Prediction & Relation Attention
        stock_preds = self.inter_fc(emb).view(b, s, 7)
        enhanced = self.relation_layer(stock_preds)
        
        # Final Classification
        return self.final_fc(enhanced.view(b, -1))