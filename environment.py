import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2

# =============================================================================
# 1. 원본 Pygame 게임 로직 (수정됨)
# =============================================================================
class Game:
    def __init__(self):
        # Pygame 모듈만 초기화 (화면 생성 X)
        pygame.init()
        self.screen = None
        self.screen_width, self.screen_height = 800, 600
        self.colors = {
            'background': (240, 220, 200), 'dough': (210, 180, 140),
            'template': (0, 0, 0, 100), 'text': (50, 50, 50)
        }
        self.font = None
        self.levels = [{'goal': 90, 'time': 30, 'templates': [{'shape': 'circle', 'center': (400, 300), 'radius': 150}]}]
        self.current_level = 0
        self.start_time = 0
        self.brush_size = 'medium'
        self.brush_radii = {'small': 15, 'medium': 30, 'large': 50}
        self.dough_surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        self.template_surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        self.completion_score = 0
        self.messiness_score = 0
        self.final_score = 0

    def init_render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Cookie Master RL")
            self.font = pygame.font.SysFont('Arial', 24)

    def start_level(self, level_index):
        self.current_level = level_index
        level_data = self.levels[self.current_level]
        self.current_level_time = level_data['time']
        self.start_time = pygame.time.get_ticks()
        self.dough_surf.fill((0, 0, 0, 0))
        self.template_surf.fill((0, 0, 0, 0))
        for t in level_data['templates']:
            if t['shape'] == 'circle':
                pygame.draw.circle(self.template_surf, self.colors['template'], t['center'], t['radius'])

    def stamp_dough_particle(self, pos):
        radius = self.brush_radii[self.brush_size]
        pygame.draw.circle(self.dough_surf, self.colors['dough'], pos, radius)

    def calculate_results(self):
        dough_mask = pygame.mask.from_surface(self.dough_surf)
        template_mask = pygame.mask.from_surface(self.template_surf)
        template_area = template_mask.count()
        if template_area > 0:
            self.completion_score = (template_mask.overlap_area(dough_mask, (0, 0)) / template_area) * 100
            mess_area = dough_mask.overlap_area(template_mask.copy().invert(), (0, 0))
            self.messiness_score = (mess_area / template_area) * 100
        else:
            self.completion_score = 0
            self.messiness_score = 0
        self.final_score = self.completion_score - self.messiness_score

    def draw(self):
        if self.screen and self.font:
            self.screen.fill(self.colors['background'])
            self.screen.blit(self.template_surf, (0, 0))
            self.screen.blit(self.dough_surf, (0, 0))
            time_left = max(0, self.current_level_time - (pygame.time.get_ticks() - self.start_time) / 1000)
            score_text = f"Completion: {self.completion_score:.1f}% | Mess: {self.messiness_score:.1f}%"
            time_text = f"Time Left: {time_left:.1f}s"
            self.screen.blit(self.font.render(score_text, True, self.colors['text']), (10, 10))
            self.screen.blit(self.font.render(time_text, True, self.colors['text']), (10, 40))

# =============================================================================
# 2. Gymnasium 환경 래퍼 클래스 (수정됨)
# =============================================================================
class CookieMasterEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = Game()
        self.screen_width = self.game.screen_width
        self.screen_height = self.game.screen_height
        self.render_mode = render_mode

        # 가상 마우스 위치 변수 추가
        self.mouse_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=np.float32)

        if self.render_mode == "human":
            self.game.init_render()

        self.observation_space = spaces.Dict({
            'dough_mask': spaces.Box(low=0, high=1, shape=(84, 84), dtype=np.float32),
            'template_mask': spaces.Box(low=0, high=1, shape=(84, 84), dtype=np.float32),
            'time_left': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'brush_size': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'mouse_pos': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(8)

    def _get_observation(self):
        dough_arr = pygame.surfarray.array3d(self.game.dough_surf)
        template_arr = pygame.surfarray.array3d(self.game.template_surf)
        dough_gray = cv2.cvtColor(dough_arr, cv2.COLOR_RGB2GRAY)
        template_gray = cv2.cvtColor(template_arr, cv2.COLOR_RGB2GRAY)
        dough_resized = cv2.resize(dough_gray, (84, 84), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        template_resized = cv2.resize(template_gray, (84, 84), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        time_left = max(0, self.game.current_level_time - (pygame.time.get_ticks() - self.game.start_time) / 1000)
        norm_time = np.array([time_left / self.game.current_level_time], dtype=np.float32)
        brush_map = {'small': 0.0, 'medium': 0.5, 'large': 1.0}
        norm_brush = np.array([brush_map[self.game.brush_size]], dtype=np.float32)
        
        # 실제 마우스 위치 대신 내부 변수 사용
        norm_mouse = self.mouse_pos / np.array([self.screen_width, self.screen_height], dtype=np.float32)

        return {
            'dough_mask': dough_resized, 'template_mask': template_resized,
            'time_left': norm_time, 'brush_size': norm_brush, 'mouse_pos': norm_mouse
        }

    def _calculate_reward(self, action):
        reward = 0
        completion_delta = self.game.completion_score - self.last_completion
        reward += completion_delta * 1.0
        reward -= (self.game.messiness_score / 100.0) * 0.1
        reward -= 0.01
        if action == 0:
            reward -= 0.05
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        if terminated or truncated:
            final_score = self.game.final_score
            goal = self.game.levels[self.game.current_level]['goal']
            if final_score >= goal:
                reward += 100.0
            else:
                reward -= 50.0
        self.last_completion = self.game.completion_score
        return float(reward)

    def _is_terminated(self):
        time_elapsed = (pygame.time.get_ticks() - self.game.start_time) / 1000
        return time_elapsed >= self.game.current_level_time

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.start_level(0)
        self.current_step = 0
        self.max_steps = 500
        
        # 가상 마우스 위치 초기화
        self.mouse_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=np.float32)

        self.game.calculate_results()
        self.last_completion = self.game.completion_score
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        move_dist = 30
        
        # 실제 마우스 대신 내부 변수 업데이트
        if action == 0:
            # Numpy 배열을 정수 튜플로 변환하여 전달
            self.game.stamp_dough_particle(tuple(self.mouse_pos.astype(int)))
        elif 1 <= action <= 4:
            if action == 1: self.mouse_pos[1] -= move_dist
            elif action == 2: self.mouse_pos[1] += move_dist
            elif action == 3: self.mouse_pos[0] -= move_dist
            elif action == 4: self.mouse_pos[0] += move_dist
            self.mouse_pos[0] = np.clip(self.mouse_pos[0], 0, self.screen_width)
            self.mouse_pos[1] = np.clip(self.mouse_pos[1], 0, self.screen_height)
        elif 5 <= action <= 7:
            self.game.brush_size = ['small', 'medium', 'large'][action - 5]

        self.game.calculate_results()
        reward = self._calculate_reward(action)
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        observation = self._get_observation()

        if self.render_mode == "human":
            # 렌더링 시에만 실제 마우스 위치를 가상 위치와 동기화
            pygame.mouse.set_pos(self.mouse_pos)
            self.render()

        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            self.game.draw()
            pygame.display.flip()

    def close(self):
        if self.game.screen is not None:
            pygame.quit()