import os
import time
from stable_baselines3 import PPO
from environment import CookieMasterEnv

def main():
    """학습된 모델을 로드하고 10개의 에피소드 동안 실행하여 성능을 시각적으로 검증합니다."""
    
    # 모델 경로 및 에피소드 수 설정
    MODEL_PATH = "models/final_model.zip"  # 경로 수정
    NUM_EPISODES = 10

    # 모델 파일 존재 여부 확인
    if not os.path.exists(MODEL_PATH):
        print(f"오류: 모델 파일을 찾을 수 없습니다. 경로: {MODEL_PATH}")
        return

    # 환경 생성 및 모델 로드
    env = CookieMasterEnv()
    model = PPO.load(MODEL_PATH, env=env)

    print(f"'{MODEL_PATH}' 모델 로드 완료. {NUM_EPISODES}개의 에피소드 동안 평가를 시작합니다.")

    try:
        for i in range(NUM_EPISODES):
            obs, info = env.reset()
            terminated, truncated = False, False
            total_reward = 0.0
            
            print(f"\n--- 에피소드 {i + 1} 시작 ---")

            while not terminated and not truncated:
                # 모델을 사용하여 행동 예측 (deterministic=True로 설정하여 가장 확률이 높은 행동 선택)
                action, _states = model.predict(obs, deterministic=True)
                
                # 환경에서 행동 실행
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 총 보상 누적
                total_reward += reward
                
                # 환경 렌더링
                env.render()

                # 시각적 확인을 위해 약간의 지연 추가
                time.sleep(0.01)

            print(f"에피소드 {i + 1} 종료. 총 보상: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\n사용자에 의해 평가가 중단되었습니다.")
    finally:
        # 환경 종료
        env.close()
        print("\n평가 완료. 환경이 종료되었습니다.")

if __name__ == "__main__":
    main()