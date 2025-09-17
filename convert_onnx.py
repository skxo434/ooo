# convert_onnx.py
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
import os, sys

# --- 중요 설정 ---
# train_simple.py, index.html과 동일한 관찰 공간 구조
OBSERVATION_SPACE_SHAPES = {
    'dough_mask':    (1, 84, 84),
    'template_mask': (1, 84, 84),
    'time_left':     (1,),
    'brush_size':    (1,),
    'mouse_pos':     (2,)
}
# -----------------

class OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        # MultiInputPolicy는 features_extractor와 mlp_extractor로 구성됩니다.
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, dough_mask, template_mask, time_left, brush_size, mouse_pos):
        # SB3 MultiInputPolicy의 순전파 로직을 그대로 구현
        observation = {
            'dough_mask': dough_mask,
            'template_mask': template_mask,
            'time_left': time_left,
            'brush_size': brush_size,
            'mouse_pos': mouse_pos
        }
        # 1. Feature Extractor
        features = self.features_extractor(observation)
        # 2. MLP Extractor
        latent_pi, _ = self.mlp_extractor(features)
        # 3. Action Net
        # .predict() 대신, 결정론적 행동을 위해 분포의 평균(mean)을 사용합니다.
        return self.action_net(latent_pi).argmax(dim=1)

def main(source_path, output_path):
    if not os.path.exists(source_path):
        print(f"오류: {source_path}에서 원본 모델을 찾을 수 없습니다.")
        return
        
    # MultiInputPolicy를 사용하는 모델 로드
    model = PPO.load(source_path, device='cpu')
    onnxable_model = OnnxablePolicy(model.policy)

    # Dict 관찰 공간에 맞는 더미 입력 생성
    dummy_input = tuple(
        torch.randn(1, *shape) for shape in OBSERVATION_SPACE_SHAPES.values()
    )
    
    # index.html에서 사용하는 입력 이름과 일치시킵니다.
    # SB3 내부 순서에 따라 이름이 매겨집니다.
    input_names = [
        "input.1", # dough_mask
        "input.4", # template_mask
        "input.7", # time_left
        "input.10", # brush_size
        "input.13" # mouse_pos
    ]

    torch.onnx.export(
        onnxable_model, 
        dummy_input, 
        output_path, 
        opset_version=12,
        input_names=input_names, 
        output_names=["output"] # index.html과 일치
    )
    print(f"최종 모델 변환 완료! '{output_path}' 파일이 생성되었습니다.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("사용법: python convert_onnx.py <입력_zip_경로> <출력_onnx_경로>")
        print("예시: python convert_onnx.py models/latest_model.zip model.onnx")
    else:
        main(sys.argv[1], sys.argv[2])
