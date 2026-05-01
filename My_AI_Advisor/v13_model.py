import torch
import torch.nn as nn

class MetaLabelingNet(nn.Module):
    """
    [V13.8 아키텍처] Meta-Labeling AI 모델 (The Gatekeeper)
    - 입력: 13차원 (5 고전 시그널 + 8 매크로 지표)
    - 출력: 0.0 ~ 1.0 (매수 승인 확률)
    """
    def __init__(self, input_dim=13, hidden_dim=128):
        super(MetaLabelingNet, self).__init__()
        
        # 딥러닝 망 구성: 배치 정규화(BatchNorm)와 드롭아웃(Dropout 0.3)으로 과적합 방지
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # V13.8의 핵심: 노이즈 대응력 강화
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 최종 출력층: Sigmoid를 통과하여 확률값으로 반환
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 입력 텐서가 1D 배열(단일 종목 추론)로 들어올 경우, BatchNorm을 위해 2D(배치 사이즈 1)로 변환
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        output = self.net(x)
        return output.squeeze(-1) # 스칼라 값으로 압축하여 반환

# (테스트용 코드 - 실제 배포 시에는 주석 처리 또는 무시됨)
if __name__ == "__main__":
    # V13_model 구조 테스트
    dummy_input = torch.randn(1, 13) # 1개의 배치, 13차원 데이터 가정
    model = MetaLabelingNet(input_dim=13)
    
    # 평가 모드로 전환 (BatchNorm, Dropout 등 비활성화)
    model.eval()
    
    with torch.no_grad():
        dummy_output = model(dummy_input)
        
    print(f"V13_Model Test Output: {dummy_output.item():.4f}")
    print("✅ V13.8 모델 아키텍처 이상 없음.")
