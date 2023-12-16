'''
    功能：
        完成了五子棋的各项基本操作，包括落子、判胜、棋盘状态更新、获取等功能
'''

import numpy as np


class ChessBoard(object):
    BLACK = 0
    WHITE = 1
    EMPTY = 2
    DIR = [[1, 0], [0, 1], [1, 1], [1, -1]]
    # 落子的奖励
    common_reward = -20  # 相同子
    draw_reward = 0     # 落子
    equal_reward = 200  # 和棋
    win_reward = 1000  # 胜利
    # 棋盘大小
    SIZE = 15

    def __init__(self):
        self.num = 0
        self.turn = self.BLACK
        self.winner = self.EMPTY
        self.premove = [-1, -1]
        self.board = np.zeros([self.SIZE, self.SIZE])
        self.board[:, :] = self.EMPTY
        self.dir = [[(-1, 0), (1, 0)], [(0, -1), (0, 1)],
                    [(-1, 1), (1, -1)], [(-1, -1), (1, 1)]]

    def printChess(self):
        print("  ", end="")
        for i in range(self.SIZE):
            print("%3d" % (i), end="")
        print("")

        for i in range(self.SIZE):
            print("%2d" % (i), end="")
            for j in range(self.SIZE):
                if self.board[i, j] == 0:
                    print("  *", end="")
                elif self.board[i, j] == 1:
                    print("  o", end="")
                else:
                    print("  -", end="")
            print("")

    def judge_Legal(self, x, y):
        return (x >= 0 and x < self.SIZE) and (y >= 0 and y < self.SIZE)

    def board(self):
        return self.board

    def draw_XY(self, x, y):
        # 判断结束了
        if x == -1 or y == -1:
            return self.board, self.common_reward, True, {}

        # 非法落子 返回-20
        if (not self.judge_Legal(x, y)):
            return self.board, self.common_reward, False, {}

        self.board[x][y] = self.turn
        self.num += 1
        self.turn = self.turn ^ 1  # 更换落子方
        self.premove = (x, y)
        winner = self.judge_Win()

        """
            这里需要格外注意!!   原代码这里说明的不清晰
            True、False到底代表对局结束还是没结束
        """
        if winner == self.EMPTY:  # 没有胜利方
            return self.board, self.draw_reward, False, {}
        else:  # 有胜利方
            return self.board, self.win_reward, True, {}

    # 判断胜利
    def judge_Win(self):
        x = 0
        y = 0
        cnt = 0
        color = self.EMPTY
        # 遍历四个方向
        for d in range(4):
            color = self.EMPTY
            cnt = 0
            # 遍历9颗连续棋子
            for k in range(-4, 5):
                x = self.premove[0] + self.DIR[d][0] * k
                y = self.premove[1] + self.DIR[d][1] * k
                if self.judge_Legal(x, y):
                    if self.board[x][y] == self.EMPTY:
                        color = self.EMPTY
                        cnt = 0
                    else:
                        if self.board[x][y] == color:
                            cnt += 1
                        else:
                            color = self.board[x][y]
                            cnt = 1
                else:
                    if k > 0:
                        break
                if cnt == 5:
                    self.winner = color
                    return color
        return self.EMPTY

    def reset(self):
        self.num = 0
        self.turn = self.BLACK
        self.winner = self.EMPTY
        self.board[:, :] = self.EMPTY
