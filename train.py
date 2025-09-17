import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from environment import CookieMasterEnv
from tqdm.auto import tqdm

class ProgressBarCallback(BaseCallback):
    """
    학습 중 프로그레스 바를 표시하기 위한 커스텀 콜백.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.pbar = None

    def _on_training_start(self) -> None:
        """학습 시작 시 호출됩니다."""
        # 총 타임스텝으로 프로그레스 바 초기화
        self.pbar = tqdm(total=self.locals['total_timesteps'], desc="Training Progress")

    def _on_step(self) -> bool:
        """매 스텝마다 호출됩니다."""
        # 프로그레스 바를 1만큼 업데이트
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        """학습 종료 시 호출됩니다."""
        # 프로그레스 바 닫기
        self.pbar.close()
        self.pbar = None

def get_latest_checkpoint(checkpoint_dir):
    """지정된 디렉토리에서 가장 최근의 체크포인트 파일 경로를 반환합니다."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("cookie_master_model_") and f.endswith(".zip")]
    if not checkpoints:
        return None

    try:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-2]))
        return os.path.join(checkpoint_dir, latest_checkpoint)
    except (ValueError, IndexError):
        return None

def setup_ppo_model(env, tensorboard_log_dir):
    """새로운 PPO 모델을 설정하고 반환합니다."""
    return PPO(
        "MultiInputPolicy",
        env,
        verbose=0, # 프로그레스 바를 사용하므로 기존 로그는 끔
        tensorboard_log=tensorboard_log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )

def main():
    checkpoint_dir = "checkpoints"
    tensorboard_dir = "tensorboard_logs"
    final_model_dir = "models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    env = Monitor(CookieMasterEnv())

    # 콜백 리스트 생성
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="cookie_master_model"
    )
    progress_callback = ProgressBarCallback()
    callbacks = [checkpoint_callback, progress_callback]

    model = None
    initial_timesteps = 0

    try:
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        
        if latest_checkpoint:
            print(f"체크포인트에서 학습 재개: {latest_checkpoint}")
            model = PPO.load(latest_checkpoint, env=env, tensorboard_log=tensorboard_dir)
            try:
                initial_timesteps = int(os.path.basename(latest_checkpoint).split('_')[-2])
            except (ValueError, IndexError):
                print("경고: 체크포인트 파일명에서 타임스텝을 파싱할 수 없습니다. 0으로 설정합니다.")
                initial_timesteps = 0
        else:
            print("새로운 학습 세션 시작")
            model = setup_ppo_model(env, tensorboard_dir)
            initial_timesteps = 0

        total_timesteps = 1_000_000
        
        # reset_num_timesteps를 True로 설정하여 프로그레스 바가 0부터 시작하도록 함
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks, # 콜백 리스트 전달
            reset_num_timesteps=True,
            tb_log_name="PPO"
        )
            
        final_model_path = os.path.join(final_model_dir, "final_model.zip")
        model.save(final_model_path)
        print(f"\n학습 완료! 최종 모델 저장됨: {final_model_path}")

    except KeyboardInterrupt:
        if model:
            # 프로그레스 바가 닫히도록 처리
            if progress_callback.pbar:
                progress_callback.pbar.close()
            print("\n학습 중단됨. 현재 모델 저장 중...")
            interrupted_model_path = os.path.join(final_model_dir, "interrupted_model.zip")
            model.save(interrupted_model_path)
            print(f"중단된 모델 저장됨: {interrupted_model_path}")
        else:
            print("\n학습 시작 전 중단됨.")
    
    finally:
        env.close()

if __name__ == "__main__":
    main()