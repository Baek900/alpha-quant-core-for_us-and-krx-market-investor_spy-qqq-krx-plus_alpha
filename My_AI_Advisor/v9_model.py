# v9_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    'num_sectors': 10,
    'hidden_dim': 64,
    'attention_heads': 4,
    'dropout_rate': 0.2
}

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True)
            self.var = torch.var(x, dim=1, keepdim=True, unbiased=False)
            return (x - self.mean) / torch.sqrt(self.var + self.eps)
        elif mode == 'denorm':
            return x * torch.sqrt(self.var[:, 0, 0] + self.eps) + self.mean[:, 0, 0]

class Hybrid_MTF_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.revin = RevIN(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.linear_ar = nn.Linear(input_dim, hidden_dim) 
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x_norm = self.revin(x, mode='norm')
        lstm_out, _ = self.lstm(x_norm)
        non_linear = lstm_out[:, -1, :] 
        linear = self.linear_ar(x_norm[:, -1, :]) 
        out = self.layer_norm(non_linear + linear)
        return out, self.revin

class V9_AI_Core(nn.Module):
    def __init__(self, sector_feature_dim, macro_latent_dim):
        super().__init__()
        self.daily_enc = Hybrid_MTF_Encoder(sector_feature_dim, CONFIG['hidden_dim'])
        self.weekly_enc = Hybrid_MTF_Encoder(sector_feature_dim, CONFIG['hidden_dim'])
        self.monthly_enc = Hybrid_MTF_Encoder(sector_feature_dim, CONFIG['hidden_dim'])
        
        combined_dim = CONFIG['hidden_dim'] * 3 
        self.cross_attention = nn.MultiheadAttention(combined_dim, CONFIG['attention_heads'], batch_first=True)
        
        self.alpha_head = nn.Sequential(
            nn.Linear(combined_dim, 32), nn.LeakyReLU(0.1),
            nn.Dropout(CONFIG['dropout_rate']), nn.Linear(32, 1)
        )
        
        self.macro_scorer = nn.Sequential(
            nn.Linear(macro_latent_dim, 64), nn.LeakyReLU(0.1),
            nn.Linear(64, CONFIG['num_sectors']), nn.Tanh()
        )
        self.regime_classifier = nn.Sequential(
            nn.Linear(macro_latent_dim, 32), nn.LeakyReLU(0.1),
            nn.Linear(32, 3), nn.Softmax(dim=-1)
        )

    def forward(self, x_daily, x_weekly, x_monthly, macro_latent):
        sector_features, revin_objects = [], []
        for i in range(CONFIG['num_sectors']):
            d_feat, revin_d = self.daily_enc(x_daily[:, i, :, :])
            w_feat, _ = self.weekly_enc(x_weekly[:, i, :, :])
            m_feat, _ = self.monthly_enc(x_monthly[:, i, :, :])
            sector_features.append(torch.cat([d_feat, w_feat, m_feat], dim=-1).unsqueeze(1))
            revin_objects.append(revin_d)
            
        sector_tensor = torch.cat(sector_features, dim=1) 
        attn_output, attn_weights = self.cross_attention(sector_tensor, sector_tensor, sector_tensor)
        
        raw_alphas = self.alpha_head(attn_output).squeeze(-1) 
        expert_alphas = torch.zeros_like(raw_alphas)
        for i in range(CONFIG['num_sectors']):
            expert_alphas[:, i] = revin_objects[i](raw_alphas[:, i], mode='denorm')
            
        macro_scores = self.macro_scorer(macro_latent) 
        regime_probs = self.regime_classifier(macro_latent) 
        return expert_alphas, macro_scores, regime_probs, attn_weights
