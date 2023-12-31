#作者：陈冠旭 朱奕坤 张方杰 吕泓泰
#项目：基于人工智能原理的五子棋及其拓展
import pygame
from tkinter import messagebox
from pygame.locals import *
from GameMap import *
from AlphaBeta import *
from EasyAI import *
from MyMnistWindow import *
from DQN_AI import *
from EasyAI import *

# 按钮功能模块


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
            self.msg_image = self.font.render(
                self.text, True, self.text_color, self.button_color[0])
        else:
            self.msg_image = self.font.render(
                self.text, True, self.text_color, self.button_color[1])
        self.msg_image_rect = self.msg_image.get_rect()
        self.msg_image_rect.center = self.rect.center

    def draw(self):  # 按钮绘制
        if self.enable:
            self.screen.fill(self.button_color[0], self.rect)
        else:
            self.screen.fill(self.button_color[1], self.rect)
        self.screen.blit(self.msg_image, self.msg_image_rect)


class StartWhiteButton(Button):  # 选取白方
    def __init__(self, screen, text, x, y):  # 构造函数
        super().__init__(screen, text, x, y, [
            (26, 173, 25), (158, 217, 157)], True)

    def click(self, game):  # 点击，pygame内置方法
        if self.enable:  # 启动游戏并初始化，变换按钮颜色
            game.start()
            game.winner = None
            game.multiple = False
            game.mode = 0
            self.msg_image = self.font.render(
                self.text, True, self.text_color, self.button_color[1])
            self.enable = False
            return True
        return False

    def unclick(self):  # 取消点击
        if not self.enable:
            self.msg_image = self.font.render(
                self.text, True, self.text_color, self.button_color[0])
            self.enable = True


class StartBlackButton(Button):  # 选取黑方
    def __init__(self, screen, text, x, y):
        super().__init__(screen, text, x, y, [
            (26, 173, 25), (158, 217, 157)], True)

    def click(self, game):
        if self.enable:  # 启动游戏并初始化，变换按钮颜色，安排AI先手
            game.start()
            game.winner = None
            game.multiple = False
            game.mode = 1
            game.useAI = True
            self.msg_image = self.font.render(
                self.text, True, self.text_color, self.button_color[1])
            self.enable = False
            return True
        return False

    def unclick(self):
        if not self.enable:
            self.msg_image = self.font.render(
                self.text, True, self.text_color, self.button_color[0])
            self.enable = True


class GiveupButton(Button):  # 投降按钮 任何模式都能用
    def __init__(self, screen, text, x, y):
        super().__init__(screen, text, x, y, [
            (153, 51, 250), (221, 160, 221)], False)  # 紫色

    def click(self, game):  # 结束游戏，判断赢家
        if self.enable:
            game.is_play = False
            pygame.mixer.music.stop()
            if game.winner is None:
                game.winner = game.map.reverseTurn(game.player)
            self.msg_image = self.font.render(
                self.text, True, self.text_color, self.button_color[1])
            self.enable = False
            return True
        return False

    def unclick(self):  # 取消点击
        if not self.enable:
            self.msg_image = self.font.render(
                self.text, True, self.text_color, self.button_color[0])
            self.enable = True


class MultiStartButton(Button):  # 开始按钮（多人游戏）
    def __init__(self, screen, text, x, y):  # 构造函数
        super().__init__(screen, text, x, y, [
            (230, 67, 64), (236, 139, 137)], True)

    def click(self, game):  # 点击，pygame内置方法
        if self.enable:  # 启动游戏并初始化，变换按钮颜色
            game.start()
            game.winner = None
            game.multiple = True
            game.mode = 2
            self.msg_image = self.font.render(
                self.text, True, self.text_color, self.button_color[1])
            self.enable = False
            return True
        return False

    def unclick(self):  # 取消点击
        if not self.enable:
            self.msg_image = self.font.render(
                self.text, True, self.text_color, self.button_color[0])
            self.enable = True


class WritingButton(Button):  # 手写按钮
    def __init__(self, screen, text, x, y):
        super().__init__(screen, text, x, y, [
            (15, 151, 50), (133, 130, 201)], False)

    def click(self, game):  # 结束游戏，判断赢家
        if self.enable:
            x, y = handwriting_result()
            if (x < 15 and x > -1 and y < 15 and y > -1 and game.map.isEmpty(x, y)):
                game.action = (x, y)
            else:
                messagebox.showinfo('错误', '输入有误重新输入')
            self.enable = False
            return True
        return False

    def unclick(self):  # 取消点击
        if not self.enable:
            self.msg_image = self.font.render(
                self.text, True, self.text_color, self.button_color[0])
            self.enable = True


class Consult_AI_Button(Button):  # AI帮手
    def __init__(self, screen, text, x, y):
        super().__init__(screen, text, x, y, [
            (100, 101, 200), (21, 10, 21)], False)

    def click(self, game):  # 给出一个提示
        use_model = False
        if self.enable:
            if (use_model == True):
                x, y = DQN_AI_value(game.map.map, game.player)
            else:
                AI = EasyAI(15)
                x, y = AI.findBestChess(game.map.map, game.player)
            messagebox.showinfo('提示', '建议下在('+str(x)+','+str(y)+')处')
            return True
        return False

    def unclick(self):  # 取消点击
        if not self.enable:
            self.msg_image = self.font.render(
                self.text, True, self.text_color, self.button_color[0])
            self.enable = True


class Game():
    def __init__(self, caption):
        pygame.init()
        pygame.mixer.init()
        messagebox.showinfo('说明', '游戏说明\n 规则：基本五子棋规则\n 模式：AI对弈，人人对弈')
        self.play_chess_sound = pygame.mixer.Sound(
            './Music/playchess.wav')  # 落子音效
        self.play_chess_sound.set_volume(2)
        self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.buttons = []
        self.buttons.append(StartWhiteButton(
            self.screen, 'choose White', MAP_WIDTH + 30, BUTTON_HEIGHT + 45))
        self.buttons.append(StartBlackButton(
            self.screen, 'choose Black', MAP_WIDTH + 30, 2*(BUTTON_HEIGHT + 45)))
        self.buttons.append(MultiStartButton(
            self.screen, 'PVP', MAP_WIDTH + 30, 3*(BUTTON_HEIGHT + 45)))
        self.buttons.append(GiveupButton(
            self.screen, 'Give Up', MAP_WIDTH + 30, 4*(BUTTON_HEIGHT + 45)))
        self.buttons.append(WritingButton(
            self.screen, 'Writing', MAP_WIDTH + 30, 5*(BUTTON_HEIGHT + 45)))
        self.buttons.append(Consult_AI_Button(
            self.screen, 'Consult AI', MAP_WIDTH + 30, 6*(BUTTON_HEIGHT + 45)))
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
        pygame.mixer.music.load('./Music/BGM.mp3')  # 背景音乐
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play()
        self.player = MAP_ENTRY_TYPE.MAP_PLAYER_ONE
        self.map.reset()

    def play(self):  # 画底板
        self.clock.tick(60)

        light_yellow = (247, 238, 214)
        pygame.draw.rect(self.screen, light_yellow,
                         pygame.Rect(0, 0, MAP_WIDTH, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(
            MAP_WIDTH, 0, INFO_WIDTH, SCREEN_HEIGHT))

        for button in self.buttons:  # 画按钮
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

    def changeMouseShow(self):  # 开始游戏的时候把鼠标预览切换成预览棋子的样子
        map_x, map_y = pygame.mouse.get_pos()
        x, y = self.map.MapPosToIndex(map_x, map_y)
        if self.map.isInMap(map_x, map_y) and self.map.isEmpty(x, y):  # 在棋盘内且当前无棋子
            pygame.mouse.set_visible(False)
            light_red = (213, 90, 107)
            pos, radius = (map_x, map_y), CHESS_RADIUS
            pygame.draw.circle(self.screen, light_red, pos, radius)
        else:
            pygame.mouse.set_visible(True)

    def checkClick(self, x, y, isAI=False):  # 后续处理
        self.map.click(x, y, self.player)
        if self.AI.isWin(self.map.map, self.player):
            self.winner = self.player
            pygame.mixer.music.stop()
            self.click_button(self.buttons[self.mode])
            if self.winner == MAP_ENTRY_TYPE.MAP_PLAYER_ONE:
                messagebox.showinfo('游戏结束', '白方胜利！')
            else:
                messagebox.showinfo('游戏结束', '黑方胜利！')
        else:
            self.player = self.map.reverseTurn(self.player)
            if not isAI:
                self.useAI = True

    def mouseClick(self, map_x, map_y):  # 处理下棋动作
        if self.is_play and self.map.isInMap(map_x, map_y) and self.winner == None:
            x, y = self.map.MapPosToIndex(map_x, map_y)
            self.play_chess_sound.play()
            if self.map.isEmpty(x, y):
                self.action = (x, y)

    def showWinner(self):  # 输出胜者
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

    def click_button(self, button):  # 点击按钮
        if button.click(self):
            for tmp in self.buttons:
                if tmp != button:
                    tmp.unclick()

    def check_buttons(self, mouse_x, mouse_y):  # 检测鼠标与按钮
        for button in self.buttons:
            if button.rect.collidepoint(mouse_x, mouse_y):
                self.click_button(button)
                break


game = Game("FIVE CHESS " + GAME_VERSION)
while True:
    game.play()
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # 退出游戏
            pygame.quit()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:  # 按下按钮
            mouse_x, mouse_y = pygame.mouse.get_pos()
            game.mouseClick(mouse_x, mouse_y)
            game.check_buttons(mouse_x, mouse_y)
