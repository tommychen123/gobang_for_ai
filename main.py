import pygame
#import easygui
from tkinter import messagebox
from pygame.locals import *
from GameMap import *
from AlphaBeta import *
from EasyAI import *


class Button():
	def __init__(self, screen, text, x, y, color, enable):
		self.screen = screen
		self.width = BUTTON_WIDTH
		self.height = BUTTON_HEIGHT
		self.button_color = color
		self.text_color = (255, 255, 255)
		self.enable = enable
		self.font = pygame.font.SysFont(None, BUTTON_HEIGHT*2//3)
		
		self.rect = pygame.Rect(0, 0, self.width, self.height)
		self.rect.topleft = (x, y)
		self.text = text
		self.init_msg()
		
	def init_msg(self):
		if self.enable:
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
		else:
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
		self.msg_image_rect = self.msg_image.get_rect()
		self.msg_image_rect.center = self.rect.center
		
	def draw(self):
		if self.enable:
			self.screen.fill(self.button_color[0], self.rect)
		else:
			self.screen.fill(self.button_color[1], self.rect)
		self.screen.blit(self.msg_image, self.msg_image_rect)
		

class StartWhiteButton(Button): # 白方
	def __init__(self, screen, text, x, y):# 构造函数
		super().__init__(screen, text, x, y, [(26, 173, 25),(158, 217, 157)], True)
	
	def click(self, game): # 点击，pygame内置方法
		if self.enable:  # 启动游戏并初始化，变换按钮颜色
			game.start()
			game.winner = None
			game.multiple = False
			game.mode=0
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
			self.enable = False
			return True
		return False
	
	def unclick(self): # 取消点击
		if not self.enable:
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
			self.enable = True

class StartBlackButton(Button): # 黑方
	def __init__(self, screen, text, x, y):
		super().__init__(screen, text, x, y, [(26, 173, 25),(158, 217, 157)], True)
	
	def click(self, game):
		if self.enable: # 启动游戏并初始化，变换按钮颜色，安排AI先手
			game.start()
			game.winner = None
			game.multiple = False
			game.mode=1
			game.useAI = True
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
			self.enable = False
			return True
		return False
	
	def unclick(self):
		if not self.enable:
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
			self.enable = True

class GiveupButton(Button):  # 投降按钮 任何模式都能用
	def __init__(self, screen, text, x, y):
		super().__init__(screen, text, x, y, [(153, 51, 250),(221, 160, 221)], False)  # 紫色
		
	def click(self, game): # 结束游戏，判断赢家
		if self.enable:
			game.is_play = False
			pygame.mixer.music.stop()
			if game.winner is None:
				game.winner = game.map.reverseTurn(game.player)
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
			self.enable = False
			return True
		return False

	def unclick(self):
		if not self.enable:
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
			self.enable = True

class MultiStartButton(Button):  # 开始按钮（多人游戏）
    def __init__(self, screen, text, x, y):  # 构造函数
        super().__init__(screen, text, x, y, [(230, 67, 64), (236, 139, 137)], True)

    def click(self, game):  # 点击，pygame内置方法
        if self.enable:  # 启动游戏并初始化，变换按钮颜色
            game.start()
            game.winner = None
            game.multiple=True
            game.mode=2
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
            self.enable = False
            return True
        return False

    def unclick(self):  # 取消点击
        if not self.enable:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
            self.enable = True

class Game():
	def __init__(self, caption):
		pygame.init()
		pygame.mixer.init()
		messagebox.showinfo( '说明','游戏说明\n 规则：基本五子棋规则\n 模式：AI对弈，人人对弈')
		self.play_chess_sound = pygame.mixer.Sound('playchess.wav') #落子音效
		self.play_chess_sound.set_volume(2)
		self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
		pygame.display.set_caption(caption)
		self.clock = pygame.time.Clock()
		self.buttons = []
		self.buttons.append(StartWhiteButton(self.screen, 'choose White', MAP_WIDTH + 30, BUTTON_HEIGHT + 45))
		self.buttons.append(StartBlackButton(self.screen, 'choose Black', MAP_WIDTH + 30, 2*(BUTTON_HEIGHT + 45)))
		self.buttons.append(MultiStartButton(self.screen, 'PVP', MAP_WIDTH + 30, 3*(BUTTON_HEIGHT + 45)))
		self.buttons.append(GiveupButton(self.screen, 'Give Up', MAP_WIDTH + 30, 4*(BUTTON_HEIGHT + 45)))
		self.is_play = False
		self.mode = 0
		self.map = Map(CHESS_LEN, CHESS_LEN)
		self.player = MAP_ENTRY_TYPE.MAP_PLAYER_ONE
		self.action = None
		self.AI = AlphaBeta(CHESS_LEN)
		self.useAI = False
		self.winner = None
		self.multiple = False
	
	def start(self):
		self.is_play = True
		pygame.mixer.music.load('BGM.mp3')
		pygame.mixer.music.set_volume(0.5)
		pygame.mixer.music.play()
		self.player = MAP_ENTRY_TYPE.MAP_PLAYER_ONE
		self.map.reset()

	def play(self):# 画底板
		self.clock.tick(60)
		
		light_yellow = (247, 238, 214)
		pygame.draw.rect(self.screen, light_yellow, pygame.Rect(0, 0, MAP_WIDTH, SCREEN_HEIGHT))
		pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(MAP_WIDTH, 0, INFO_WIDTH, SCREEN_HEIGHT))
		
		for button in self.buttons:# 画按钮
			button.draw()
		
		if self.is_play and self.winner == None:
			if self.useAI and not self.multiple:
				x, y = self.AI.findBestChess(self.map.map, self.player)
				self.checkClick(x, y, True)
				self.useAI = False

			if self.action is not None:
				self.checkClick(self.action[0], self.action[1])
				self.action = None
			
			if self.winner == None:
				self.changeMouseShow()
			
		if self.winner != None:
			self.showWinner()

		self.map.drawBackground(self.screen)
		self.map.drawChess(self.screen)

	
	def changeMouseShow(self): # 开始游戏的时候把鼠标预览切换成预览棋子的样子
		map_x, map_y = pygame.mouse.get_pos()
		x, y = self.map.MapPosToIndex(map_x, map_y)
		if self.map.isInMap(map_x, map_y) and self.map.isEmpty(x, y):# 在棋盘内且当前无棋子
			pygame.mouse.set_visible(False)
			light_red = (213, 90, 107)
			pos, radius = (map_x, map_y), CHESS_RADIUS
			pygame.draw.circle(self.screen, light_red, pos, radius)
		else:
			pygame.mouse.set_visible(True)
	
	def checkClick(self,x, y, isAI=False): # 后续处理
		self.map.click(x, y, self.player)
		if self.AI.isWin(self.map.map, self.player):
			self.winner = self.player
			pygame.mixer.music.stop()
			self.click_button(self.buttons[self.mode])
			if self.winner == MAP_ENTRY_TYPE.MAP_PLAYER_ONE:
				messagebox.showinfo( '游戏结束','白方胜利！')
			else:
				messagebox.showinfo( '游戏结束','黑方胜利！')
		else:	
			self.player = self.map.reverseTurn(self.player)
			if not isAI:	
				self.useAI = True
	
	def mouseClick(self, map_x, map_y):# 处理下棋动作
		if self.is_play and self.map.isInMap(map_x, map_y) and self.winner == None:
			x, y = self.map.MapPosToIndex(map_x, map_y)
			self.play_chess_sound.play()
			if self.map.isEmpty(x, y):
				self.action = (x, y)
	'''
	def isOver(self): # 中断条件
		print(self.winner)
		return self.winner is not None
	'''
	def showWinner(self):# 输出胜者
		def showFont(screen, text, location_x, locaiton_y, height):
			font = pygame.font.SysFont(None, height)
			font_image = font.render(text, True, (0, 0, 255), (255, 255, 255))
			font_image_rect = font_image.get_rect()
			font_image_rect.x = location_x
			font_image_rect.y = locaiton_y
			screen.blit(font_image, font_image_rect)
		if self.winner == MAP_ENTRY_TYPE.MAP_PLAYER_ONE:
			str = 'Winner is White'
		else:
			str = 'Winner is Black'
		showFont(self.screen, str, MAP_WIDTH + 25, SCREEN_HEIGHT - 60, 30)
		pygame.mouse.set_visible(True)
	
	def click_button(self, button):
		if button.click(self):
			for tmp in self.buttons:
				if tmp != button:
					tmp.unclick()
					
	def check_buttons(self, mouse_x, mouse_y):
		for button in self.buttons:
			if button.rect.collidepoint(mouse_x, mouse_y):
				self.click_button(button)
				break
			
game = Game("FIVE CHESS " + GAME_VERSION)
while True:
	game.play()
	pygame.display.update()
	
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			exit()
		elif event.type == pygame.MOUSEBUTTONDOWN:
			mouse_x, mouse_y = pygame.mouse.get_pos()
			game.mouseClick(mouse_x, mouse_y)
			game.check_buttons(mouse_x, mouse_y)
