import gymnasium as gym
from gymnasium import spaces
import asyncio
import websockets
import json
import numpy as np
import threading
import logging
import queue

logging.basicConfig(level=logging.INFO)

# --- 추가된 부분: 설정을 바탕으로 Gymnasium 공간을 재귀적으로 생성하는 헬퍼 함수 ---
def create_space_from_config(config):
    """설정 딕셔너리를 바탕으로 Gymnasium 공간 객체를 생성합니다."""
    space_type = config['type']
    
    if space_type == 'box' or space_type == 'box_vector':
        return spaces.Box(
            low=config.get('low', -np.inf),
            high=config.get('high', np.inf),
            shape=tuple(config['shape']),
            dtype=np.float32
        )
    elif space_type == 'box_image':
        # SB3 CnnPolicy는 (C, H, W)를 기본으로 하지만,
        # 일반적으로 (H, W, C)도 잘 처리합니다. 여기서는 유연하게 shape을 그대로 사용합니다.
        return spaces.Box(
            low=0,
            high=255,
            shape=tuple(config['shape']),
            dtype=np.uint8
        )
    elif space_type == 'discrete':
        return spaces.Discrete(config['n'])
    elif space_type == 'dict':
        return spaces.Dict({
            key: create_space_from_config(value)
            for key, value in config['spaces'].items()
        })
    else:
        raise ValueError(f"지원하지 않는 공간 타입입니다: {space_type}")
# --- 추가 종료 ---


class ServerThread(threading.Thread):
    """
    백그라운드에서 웹소켓 서버를 실행하고 메인 스레드와 큐를 통해 통신하는
    독립적인 스레드입니다.
    """
    def __init__(self):
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.command_queue = queue.Queue()
        self.observation_queue = queue.Queue()
        self.client_connected_event = threading.Event() # --- 추가: 동기화 이벤트 생성 ---
        self.daemon = True # 메인 스레드가 종료되면 함께 종료

    # --- FIX ---
    # 최신 websockets 라이브러리(v10.0+)는 핸들러에 'path' 인자를 전달하지 않습니다.
    # 따라서 함수 정의에서 'path'를 제거해야 합니다.
    async def handler(self, websocket):
        logging.info(f"새로운 클라이언트 연결됨: {websocket.remote_address}")
        self.client_connected_event.set() # --- 추가: 클라이언트가 연결되었음을 알림 ---

        consumer_task = asyncio.ensure_future(self.consumer(websocket))
        producer_task = asyncio.ensure_future(self.producer(websocket))
        done, pending = await asyncio.wait(
            [consumer_task, producer_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        
        self.client_connected_event.clear()
        logging.info(f"클라이언트 연결 종료: {websocket.remote_address}")

    async def consumer(self, websocket):
        """메인 스레드로부터 명령을 받아 클라이언트로 전송"""
        while True:
            try:
                # --- 수정: 루프가 닫히고 있을 때 발생하는 오류를 방지하기 위해 asyncio.Queue 사용 ---
                command = await self.async_command_queue.get()
                await websocket.send(json.dumps(command))
                self.async_command_queue.task_done()
            except asyncio.CancelledError:
                break # 태스크가 취소되면 루프 종료
            except websockets.exceptions.ConnectionClosed:
                break

    async def producer(self, websocket):
        """클라이언트로부터 관측 데이터를 받아 메인 스레드로 전송"""
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                # --- 수정: 루프가 닫히고 있을 때 발생하는 오류를 방지하기 위해 asyncio.Queue 사용 ---
                await self.async_observation_queue.put(data)
            except asyncio.CancelledError:
                break # 태스크가 취소되면 루프 종료
            except (websockets.exceptions.ConnectionClosed):
                break

    def run(self):
        """스레드의 메인 실행 함수. asyncio 이벤트 루프를 설정하고 실행합니다."""
        asyncio.set_event_loop(self.loop)
        # --- 수정: 스레드 간 통신을 위해 asyncio.Queue를 루프 내에서 생성 ---
        self.async_command_queue = asyncio.Queue()
        self.async_observation_queue = asyncio.Queue()
        
        start_server = websockets.serve(self.handler, "localhost", 8765)
        self.server = self.loop.run_until_complete(start_server)
        self.loop.run_forever()
        
        # 루프가 멈춘 후 정리 작업
        self.server.close()
        self.loop.run_until_complete(self.server.wait_closed())
        self.loop.close()
        logging.info("이벤트 루프가 완전히 종료되었습니다.")

    def stop(self):
        """서버와 이벤트 루프를 안전하게 종료합니다."""
        if not self.loop.is_running():
            return

        logging.info("웹소켓 서버 종료를 시도합니다.")
        # --- 수정: 안전한 종료 로직 ---
        # 1. 실행 중인 모든 태스크를 가져와서 취소합니다.
        tasks = asyncio.all_tasks(loop=self.loop)
        for task in tasks:
            task.cancel()

        # 2. 루프를 멈추도록 스케줄링합니다.
        self.loop.call_soon_threadsafe(self.loop.stop)
        # 3. 스레드가 완전히 종료될 때까지 기다립니다.
        self.join()
        logging.info("서버 스레드가 성공적으로 종료되었습니다.")

    # --- 추가: 메인 스레드에서 안전하게 큐에 접근하기 위한 메서드 ---
    def send_command(self, command):
        self.loop.call_soon_threadsafe(self.command_queue.put_nowait, command)

    def get_observation(self):
        return self.observation_queue.get()


class WebSocketEnv(gym.Env):
    def __init__(self, observation_space_config, action_space_config):
        super(WebSocketEnv, self).__init__()

        # --- 수정된 부분: 헬퍼 함수를 사용하여 관찰/행동 공간 생성 ---
        self.observation_space = create_space_from_config(observation_space_config)
        self.action_space = create_space_from_config(action_space_config)
        # --- 수정 종료 ---
        
        self.server_thread = ServerThread()
        self.server_thread.start()
        logging.info("클라이언트 연결 대기 중...")
        self.server_thread.client_connected_event.wait()
        logging.info("클라이언트 연결 완료.")

    def _process_observation(self, obs_data):
        """JSON으로 받은 관찰 데이터를 NumPy 배열로 변환합니다."""
        if isinstance(self.observation_space, spaces.Dict):
            # Dict 공간일 경우, 각 키에 대해 데이터를 변환합니다.
            processed_obs = {}
            for key, value in obs_data.items():
                space = self.observation_space.spaces[key]
                processed_obs[key] = np.array(value, dtype=space.dtype).reshape(space.shape)
            return processed_obs
        else:
            # 단순 Box 공간일 경우
            return np.array(obs_data, dtype=self.observation_space.dtype).reshape(self.observation_space.shape)

    # --- 수정: 큐를 사용하여 통신하는 방식으로 변경 ---
    def _send_command_and_get_response(self, command):
        """명령을 큐에 넣고, 응답 큐에서 결과를 기다립니다."""
        while not self.server_thread.observation_queue.empty():
            try:
                self.server_thread.observation_queue.get_nowait()
            except queue.Empty:
                break
        
        # --- 수정: 스레드 안전한 방식으로 명령 전송 ---
        self.server_thread.send_command(command)
        # --- 수정: 스레드 안전한 방식으로 응답 수신 ---
        response = self.server_thread.get_observation()
        return response

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        response = self._send_command_and_get_response({'command': 'reset'})
        return self._process_observation(response['observation']), {}

    def step(self, action):
        action_to_send = int(action) if isinstance(action, np.integer) else action
        
        response = self._send_command_and_get_response({'command': 'action', 'action': action_to_send})
        
        observation = self._process_observation(response['observation'])
        reward = response['reward']
        terminated = response['done']
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info

    def close(self):
        self.server_thread.stop()
        print("웹소켓 서버가 종료되었습니다.")
