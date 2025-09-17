import pygame
import sys
import random
import math
import json
import os

# --- 기본 설정 ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("쿠키 마스터 챌린지")
clock = pygame.time.Clock()
FPS = 60

# --- 파일 경로 ---
SAVE_FILE = "save_data.json"

# --- 색상 정의 ---
COLORS = {
    'BG': (245, 245, 220), 'TEMPLATE': (200, 200, 200),
    'DOUGH': (227, 197, 148), 'DOUGH_SHADOW': (201, 171, 122),
    'DOUGH_HIGHLIGHT': (242, 222, 186), 'CHOCO_CHIP': (61, 40, 20, 200),
    'TEXT': (0, 0, 0), 'SHADOW': (0, 0, 0, 50), 'WHITE': (255, 255, 255),
    'BUTTON_NORMAL': (130, 130, 130), 'BUTTON_HOVER': (160, 160, 160),
    'BUTTON_ACTIVE': (100, 100, 200), 'BUTTON_LOCKED': (80, 80, 80),
    'SUCCESS': (60, 179, 113), 'FAIL': (220, 20, 60), 'STAR': (255, 215, 0)
}

# --- 폰트 설정 ---
main_font = pygame.font.SysFont("malgungothic", 30)
title_font = pygame.font.SysFont("malgungothic", 60)
result_font = pygame.font.SysFont("malgungothic", 45)
ui_font = pygame.font.SysFont("malgungothic", 24)

# --- 도우미 함수 ---
def draw_text(text, font, color, surface, x, y, center=False):
    textobj = font.render(text, 1, color); textrect = textobj.get_rect()
    if center: textrect.center = (x, y)
    else: textrect.topleft = (x, y)
    surface.blit(textobj, textrect)

# --- UI 버튼 클래스 ---
class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect); self.text = text; self.is_hovered = False; self.scale = 1.0
    def draw(self, surface, font, color_normal, color_hover):
        mx, my = pygame.mouse.get_pos(); self.is_hovered = self.rect.collidepoint(mx, my)
        target_scale = 1.05 if self.is_hovered else 1.0; self.scale += (target_scale - self.scale) * 0.2
        scaled_width = int(self.rect.width * self.scale); scaled_height = int(self.rect.height * self.scale)
        scaled_rect = pygame.Rect(self.rect.centerx-scaled_width//2, self.rect.centery-scaled_height//2, scaled_width, scaled_height)
        color = color_hover if self.is_hovered else color_normal
        pygame.draw.rect(surface, color, scaled_rect, border_radius=10)
        draw_text(self.text, font, COLORS['WHITE'], surface, scaled_rect.centerx, scaled_rect.centery, center=True)
    def check_click(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.is_hovered

# --- 게임 메인 클래스 ---
class Game:
    def __init__(self):
        self.game_state = "main_menu"
        self.current_level = 0
        self.levels = self.define_levels()
        self.dough_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self.dough_particle_img = self.create_dough_particle_image(32)
        self.brush_size = 'medium'
        self.brush_sizes = {'small': 20, 'medium': 40, 'large': 70}
        self.load_game_data()
        self.last_mouse_pos = None
        # 무한 모드용 변수
        self.endless_score = 0
        self.endless_timer_end = 0
        self.came_from_endless = False

    def load_game_data(self):
        try:
            with open(SAVE_FILE, 'r') as f:
                self.game_data = json.load(f)
                if len(self.game_data['stars']) != len(self.levels): raise FileNotFoundError 
        except (FileNotFoundError, json.JSONDecodeError):
            self.game_data = {'stars': [0] * len(self.levels)}; self.save_game_data()
    
    def save_game_data(self):
        with open(SAVE_FILE, 'w') as f: json.dump(self.game_data, f)

    def define_levels(self):
        center = (WIDTH//2, HEIGHT//2)
        star_points = lambda c, s: [(c[0], c[1]-s), (c[0]+s*0.23, c[1]-s*0.33), (c[0]+s*0.93, c[1]-s*0.33), (c[0]+s*0.37, c[1]+s*0.13), (c[0]+s*0.57, c[1]+s*0.87), (c[0], c[1]+s*0.37), (c[0]-s*0.57, c[1]+s*0.87), (c[0]-s*0.37, c[1]+s*0.13), (c[0]-s*0.93, c[1]-s*0.33), (c[0]-s*0.23, c[1]-s*0.33)]
        return [
            {'templates': [{'shape':'circle', 'radius':100, 'center':center}], 'goal':90},
            {'templates': [{'shape':'rect', 'rect':pygame.Rect(center[0]-100, center[1]-100, 200, 200)}], 'goal':90},
            {'templates': [{'shape':'star', 'points':star_points(center, 120)}], 'goal':85},
            {'templates': [{'shape':'circle', 'radius':70, 'center':(center[0]-120, center[1])}, {'shape':'circle', 'radius':70, 'center':(center[0]+120, center[1])}], 'goal':88},
            {'templates': [{'shape':'rect', 'rect':pygame.Rect(center[0]-150, center[1]-100, 100, 200)}, {'shape':'rect', 'rect':pygame.Rect(center[0]+50, center[1]-100, 100, 200)}], 'goal':88},
            {'templates': [{'shape':'circle', 'radius':80, 'center':(center[0]-120, center[1])}, {'shape':'rect', 'rect':pygame.Rect(center[0]+40, center[1]-80, 160, 160)}], 'goal':85},
            {'templates': [{'shape':'star', 'points':star_points((center[0]+120, center[1]), 90)}, {'shape':'rect', 'rect':pygame.Rect(center[0]-210, center[1]-60, 120, 120)}], 'goal':82},
            {'templates': [{'shape':'circle', 'radius':60, 'center':(center[0], center[1]-110)}, {'shape':'circle', 'radius':60, 'center':(center[0]-110, center[1]+60)}, {'shape':'circle', 'radius':60, 'center':(center[0]+110, center[1]+60)}], 'goal':85},
            {'templates': [{'shape':'rect', 'rect':pygame.Rect(center[0]-150, center[1]-150, 100, 100)}, {'shape':'rect', 'rect':pygame.Rect(center[0]+50, center[1]-150, 100, 100)}, {'shape':'rect', 'rect':pygame.Rect(center[0]-50, center[1]+50, 100, 100)}], 'goal':85},
            {'templates': [{'shape':'circle', 'radius':50, 'center':(center[0]-150, center[1])}, {'shape':'rect', 'rect':pygame.Rect(center[0]-50, center[1]-50, 100, 100)}, {'shape':'star', 'points':star_points((center[0]+150, center[1]), 60)}], 'goal':80},
        ]
    
    def create_dough_particle_image(self, size):
        if size < 1: return None
        size = int(size)
        surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        center_pos = (size, size)
        pygame.draw.circle(surf, (*COLORS['DOUGH_SHADOW'], 150), center_pos, size)
        pygame.draw.circle(surf, COLORS['DOUGH'], center_pos, int(size * 0.9))
        pygame.draw.circle(surf, (*COLORS['DOUGH_HIGHLIGHT'], 200), (size*0.8, size*0.8), int(size*0.35))
        for _ in range(int(size*size*0.5)):
            angle = random.uniform(0, 2*math.pi); r = random.uniform(0, size*0.9)
            x, y = int(center_pos[0]+r*math.cos(angle)), int(center_pos[1]+r*math.sin(angle))
            variation = random.randint(-15, 15)
            speck_color = (max(0,min(255,COLORS['DOUGH'][0]+variation)), max(0,min(255,COLORS['DOUGH'][1]+variation)), max(0,min(255,COLORS['DOUGH'][2]+variation)))
            surf.set_at((x, y), (*speck_color, random.randint(15, 40)))
        return surf

    def start_level(self, level_index):
        self.current_level = level_index; self.game_state = "playing"
        self.reset_dough(); self.start_time = pygame.time.get_ticks()
        initial, min_t, decrement = 60, 15, 4
        self.current_level_time = max(min_t, initial - self.current_level * decrement)

    # <<<--- [핵심 추가] 무한 모드를 시작하는 함수 ---
    def start_endless_mode(self):
        self.game_state = "endless_playing"
        self.current_level = 0
        self.endless_score = 0
        self.endless_timer_end = pygame.time.get_ticks() + 60 * 1000 # 60초 타이머 설정
        self.reset_dough()

    def reset_dough(self):
        self.dough_surf.fill((0,0,0,0))
    
    # <<<--- [핵심 수정] 새로운 게임 상태를 처리하도록 run 함수 변경 ---
    def run(self):
        while True:
            if self.game_state == "main_menu": self.main_menu_loop()
            elif self.game_state == "playing": self.playing_loop()
            elif self.game_state == "endless_playing": self.endless_playing_loop()
            elif self.game_state == "results": self.results_loop()

    def main_menu_loop(self):
        start_button = Button((WIDTH/2-100, HEIGHT/2, 200, 50), "게임 시작")
        while self.game_state == "main_menu":
            screen.fill(COLORS['BG'])
            draw_text("쿠키 마스터 챌린지", title_font, COLORS['TEXT'], screen, WIDTH/2, HEIGHT/3, center=True)
            total_stars = sum(self.game_data['stars']); draw_text(f"총 획득한 별: {total_stars}", main_font, COLORS['TEXT'], screen, WIDTH/2, HEIGHT*0.9, center=True)
            start_button.draw(screen, main_font, COLORS['BUTTON_NORMAL'], COLORS['BUTTON_HOVER'])
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if start_button.check_click(event): self.start_level(0)
            pygame.display.update(); clock.tick(FPS)

    def playing_loop(self):
        # (이전과 동일)
        is_mouse_pressed, ui_interaction_active = False, False
        brush_buttons = {name: Button((WIDTH-180+i*50, 20, 40, 40), name[0].upper()) for i, name in enumerate(self.brush_sizes.keys())}
        done_button = Button((WIDTH-170, HEIGHT-70, 150, 50), "완성!"); reset_button = Button((20, HEIGHT-70, 150, 50), "초기화")
        while self.game_state == "playing":
            elapsed = (pygame.time.get_ticks()-self.start_time)/1000; remaining = max(0, self.current_level_time-elapsed)
            if remaining <= 0: self.end_level()
            current_mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    is_mouse_pressed = True; is_ui_click = False
                    if done_button.is_hovered or reset_button.is_hovered or any(b.is_hovered for b in brush_buttons.values()): is_ui_click = True
                    if is_ui_click:
                        ui_interaction_active = True
                        if done_button.check_click(event): self.end_level()
                        if reset_button.check_click(event): self.reset_dough()
                        for size, button in brush_buttons.items():
                            if button.check_click(event): self.brush_size = size
                    else: self.last_mouse_pos = current_mouse_pos; self.stamp_dough_particle(current_mouse_pos)
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1: is_mouse_pressed, ui_interaction_active, self.last_mouse_pos = False, False, None
            if is_mouse_pressed and not ui_interaction_active:
                if self.last_mouse_pos and self.last_mouse_pos != current_mouse_pos: self.draw_stretched_dough(self.last_mouse_pos, current_mouse_pos)
                self.stamp_dough_particle(current_mouse_pos); self.last_mouse_pos = current_mouse_pos
            screen.fill(COLORS['BG']); screen.blit(self.dough_surf, (0,0)); self.draw_template()
            timer_color = COLORS['FAIL'] if remaining < 10 else COLORS['TEXT']
            draw_text(f"남은 시간: {remaining:.1f}", main_font, timer_color, screen, 20, 20)
            for size, button in brush_buttons.items():
                button.draw(screen, ui_font, COLORS['BUTTON_NORMAL'], COLORS['BUTTON_HOVER'])
                if self.brush_size == size: pygame.draw.rect(screen, COLORS['BUTTON_ACTIVE'], button.rect, 3, 5)
            done_button.draw(screen, main_font, COLORS['SUCCESS'], COLORS['BUTTON_HOVER'])
            reset_button.draw(screen, main_font, COLORS['FAIL'], COLORS['BUTTON_HOVER'])
            pygame.display.update(); clock.tick(FPS)

    # <<<--- [핵심 추가] 무한 모드 전용 게임 루프 ---
    def endless_playing_loop(self):
        is_mouse_pressed, ui_interaction_active = False, False
        brush_buttons = {name: Button((WIDTH-180+i*50, 20, 40, 40), name[0].upper()) for i, name in enumerate(self.brush_sizes.keys())}
        done_button = Button((WIDTH-170, HEIGHT-70, 150, 50), "완성!"); reset_button = Button((20, HEIGHT-70, 150, 50), "초기화")
        
        while self.game_state == "endless_playing":
            remaining = max(0, (self.endless_timer_end - pygame.time.get_ticks()) / 1000)
            if remaining <= 0: self.end_level(came_from_endless=True)

            current_mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    is_mouse_pressed = True; is_ui_click = False
                    if done_button.is_hovered or reset_button.is_hovered or any(b.is_hovered for b in brush_buttons.values()): is_ui_click = True
                    if is_ui_click:
                        ui_interaction_active = True
                        if reset_button.check_click(event): self.reset_dough()
                        for size, button in brush_buttons.items():
                            if button.check_click(event): self.brush_size = size
                        if done_button.check_click(event):
                            self.calculate_results()
                            if self.final_score >= self.levels[self.current_level]['goal']:
                                self.endless_score += 1
                                self.current_level = (self.current_level + 1) % len(self.levels)
                                self.reset_dough()
                    else: self.last_mouse_pos = current_mouse_pos; self.stamp_dough_particle(current_mouse_pos)
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1: is_mouse_pressed, ui_interaction_active, self.last_mouse_pos = False, False, None
            
            if is_mouse_pressed and not ui_interaction_active:
                if self.last_mouse_pos and self.last_mouse_pos != current_mouse_pos: self.draw_stretched_dough(self.last_mouse_pos, current_mouse_pos)
                self.stamp_dough_particle(current_mouse_pos); self.last_mouse_pos = current_mouse_pos

            screen.fill(COLORS['BG']); screen.blit(self.dough_surf, (0,0)); self.draw_template()
            timer_color = COLORS['FAIL'] if remaining < 10 else COLORS['TEXT']
            draw_text(f"남은 시간: {remaining:.1f}", main_font, timer_color, screen, 20, 20)
            draw_text(f"클리어: {self.endless_score}개", main_font, COLORS['TEXT'], screen, 20, 60)
            for size, button in brush_buttons.items():
                button.draw(screen, ui_font, COLORS['BUTTON_NORMAL'], COLORS['BUTTON_HOVER'])
                if self.brush_size == size: pygame.draw.rect(screen, COLORS['BUTTON_ACTIVE'], button.rect, 3, 5)
            done_button.draw(screen, main_font, COLORS['SUCCESS'], COLORS['BUTTON_HOVER'])
            reset_button.draw(screen, main_font, COLORS['FAIL'], COLORS['BUTTON_HOVER'])
            pygame.display.update(); clock.tick(FPS)

    def results_loop(self):
        menu_button = Button((WIDTH/2-100, HEIGHT*0.7+70, 200, 50), "메인 메뉴")
        
        if self.came_from_endless:
            retry_button = Button((WIDTH/2-100, HEIGHT*0.7, 200, 50), "다시 도전")
            while self.game_state == "results":
                screen.fill(COLORS['BG'])
                draw_text("타임 오버!", title_font, COLORS['FAIL'], screen, WIDTH/2, HEIGHT/3, center=True)
                draw_text(f"최종 클리어: {self.endless_score}개", result_font, COLORS['TEXT'], screen, WIDTH/2, HEIGHT/2, center=True)
                retry_button.draw(screen, main_font, COLORS['BUTTON_NORMAL'], COLORS['BUTTON_HOVER'])
                menu_button.draw(screen, main_font, COLORS['BUTTON_NORMAL'], COLORS['BUTTON_HOVER'])
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                    if retry_button.check_click(event): self.start_endless_mode()
                    if menu_button.check_click(event): self.game_state = "main_menu"
                pygame.display.update(); clock.tick(FPS)
        else:
            next_button=Button((WIDTH/2-100,HEIGHT*0.7,200,50),"다음 단계"); retry_button=Button((WIDTH/2-100,HEIGHT*0.7,200,50),"다시 도전")
            star_rating, goal_score = 0, self.levels[self.current_level]['goal']
            if self.final_score>=goal_score: star_rating=3
            elif self.final_score>=goal_score*0.8: star_rating=2
            elif self.final_score>=goal_score*0.6: star_rating=1
            if star_rating > self.game_data['stars'][self.current_level]: self.game_data['stars'][self.current_level]=star_rating; self.save_game_data()
            success = star_rating > 0
            is_last_level = self.current_level+1 >= len(self.levels)
            if success and is_last_level: next_button.text = "무한 도전!"
            
            while self.game_state == "results":
                screen.fill(COLORS['BG'])
                result_text, color = ("성공!", COLORS['SUCCESS']) if success else ("아쉬워요!", COLORS['FAIL'])
                draw_text(result_text, title_font, color, screen, WIDTH/2, 100, center=True)
                for i in range(3):
                    star_color = COLORS['STAR'] if i<star_rating else COLORS['TEMPLATE']
                    pygame.draw.polygon(screen, star_color, self.get_star_points((WIDTH/2 - 90 + i*90, 200), 40))
                draw_text(f"최종 완성도: {self.final_score:.1f}%", result_font, COLORS['TEXT'], screen, WIDTH/2, 300, center=True)
                if success and not is_last_level: next_button.draw(screen, main_font, COLORS['BUTTON_NORMAL'], COLORS['BUTTON_HOVER'])
                elif success and is_last_level: next_button.draw(screen, main_font, COLORS['SUCCESS'], COLORS['BUTTON_HOVER'])
                else: retry_button.draw(screen, main_font, COLORS['BUTTON_NORMAL'], COLORS['BUTTON_HOVER'])
                menu_button.draw(screen, main_font, COLORS['BUTTON_NORMAL'], COLORS['BUTTON_HOVER'])
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                    if menu_button.check_click(event): self.game_state = "main_menu"
                    if success:
                        if next_button.check_click(event):
                            if is_last_level: self.start_endless_mode()
                            else: self.start_level(self.current_level+1)
                    else:
                        if retry_button.check_click(event): self.start_level(self.current_level)
                pygame.display.update(); clock.tick(FPS)
            
    def get_star_points(self, center, size):
        points=[]
        for i in range(5):
            angle=math.pi/2+2*math.pi*i/5; p1=(center[0]+size*math.cos(angle), center[1]-size*math.sin(angle)); points.append(p1)
            angle+=math.pi/5; p2=(center[0]+(size/2)*math.cos(angle), center[1]-(size/2)*math.sin(angle)); points.append(p2)
        return points

    def end_level(self, came_from_endless=False):
        if self.game_state in ["playing", "endless_playing"]:
            self.came_from_endless = came_from_endless
            if not came_from_endless: self.calculate_results()
            self.game_state = "results"
            
    def draw_stretched_dough(self, start_pos, end_pos):
        dx,dy=end_pos[0]-start_pos[0],end_pos[1]-start_pos[1]; distance=math.hypot(dx,dy)
        if distance<2: return
        angle=math.degrees(math.atan2(-dy,dx)); width=self.brush_sizes[self.brush_size]
        stretched_surf=pygame.transform.scale(self.dough_particle_img, (int(distance), width)); rotated_surf=pygame.transform.rotate(stretched_surf, angle)
        center_pos=((start_pos[0]+end_pos[0])/2, (start_pos[1]+end_pos[1])/2); rect=rotated_surf.get_rect(center=center_pos)
        shadow_img=rotated_surf.copy(); shadow_img.fill(COLORS['SHADOW'], special_flags=pygame.BLEND_RGBA_MULT)
        self.dough_surf.blit(shadow_img, (rect.x+3, rect.y+3)); self.dough_surf.blit(rotated_surf, rect)

    def stamp_dough_particle(self, pos):
        mx,my=pos; size=self.brush_sizes[self.brush_size]; angle=random.randint(0,360)
        scaled_img=pygame.transform.scale(self.dough_particle_img, (size,size)); img=pygame.transform.rotate(scaled_img, angle)
        rect=img.get_rect(center=(mx,my)); shadow_img=img.copy(); shadow_img.fill(COLORS['SHADOW'], special_flags=pygame.BLEND_RGBA_MULT)
        self.dough_surf.blit(shadow_img, (rect.x+3, rect.y+3)); self.dough_surf.blit(img, rect)
        if random.random()<0.05:
            cs=size/5; cx,cy=mx+random.randint(-int(size/3),int(size/3)), my+random.randint(-int(size/3),int(size/3))
            pygame.draw.circle(self.dough_surf, COLORS['CHOCO_CHIP'], (cx,cy), cs)

    def draw_template(self):
        for template in self.levels[self.current_level]['templates']:
            if template['shape']=='circle': pygame.draw.circle(screen, COLORS['TEMPLATE'], template['center'], template['radius'], 5)
            elif template['shape']=='rect': pygame.draw.rect(screen, COLORS['TEMPLATE'], template['rect'], 5)
            elif template['shape']=='star': pygame.draw.polygon(screen, COLORS['TEMPLATE'], template['points'], 5)

    def calculate_results(self):
        template_mask_surf=pygame.Surface((WIDTH,HEIGHT),pygame.SRCALPHA);
        for template in self.levels[self.current_level]['templates']:
            if template['shape']=='circle': pygame.draw.circle(template_mask_surf,COLORS['WHITE'],template['center'],template['radius'])
            elif template['shape']=='rect': pygame.draw.rect(template_mask_surf,COLORS['WHITE'],template['rect'])
            elif template['shape']=='star': pygame.draw.polygon(template_mask_surf,COLORS['WHITE'],template['points'])
        template_mask=pygame.mask.from_surface(template_mask_surf); dough_mask=pygame.mask.from_surface(self.dough_surf)
        total_template=template_mask.count() or 1; filled=template_mask.overlap_area(dough_mask, (0,0))
        messy=dough_mask.count()-filled
        self.completion_score=(filled/total_template)*100; self.messiness_score=(messy/total_template)*100
        self.final_score = max(0, self.completion_score - (self.messiness_score*0.5))

# --- 게임 인스턴스 생성 및 실행 ---
if __name__ == '__main__':
    game = Game()
    game.run()