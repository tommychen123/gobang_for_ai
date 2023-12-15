# In[1]
from ChessBoard import ChessBoard
import os
import math
import time
import random
import numpy as np
from collections import deque
from queue import deque
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# In[2]

EMPTY = 2
# 超参数
GAMMA = 0.9         # Q衰减系数
INITIAL_E = 0.1     # 初始ε
FINAL_E = 0.001		# 初始ε
REPLAY_SIZE = 10000  # 经验回放大小
BATCH_SIZE = 200  	# 批大小
TAGET_Q_STEP = 100  # 目标网络同步训练次数


class DQN():
    def __init__(self):
        # 棋盘大小
        self.SIZE = ChessBoard.SIZE
        self.state_dim = self.SIZE * self.SIZE
        self.action_dim = self.SIZE * self.SIZE
        # 其它参数
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = INITIAL_E

        # 创建网络
        self.create_Q()
        self.create_targetQ()
        self.targetQ_step = TAGET_Q_STEP
        # 定义优化器
        self.train_method()
        print(tf.test.is_gpu_available())
        # 定义执行器
        gpu_options = tf.GPUOptions(allow_growth=True)
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9, allow_growth=True)  # 每个gpu占用0.8的显存
        config = tf.ConfigProto(
            gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
        # 如果电脑有多个GPU，tensorflow默认全部使用。如果想只使用部分GPU，可以设置CUDA_VISIBLE_DEVICES。
        self.sess = tf.Session(config=config)

        # 网络初始化
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def create_Q(self):
        # 网络权值
        W1 = self.weight_variable([5, 5, 1, 16])
        b1 = self.bias_variable([16])  # 5*5*16
        W2 = self.weight_variable([5*5*16+1, 225])
        b2 = self.bias_variable([1, 225])

        # 输入层
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        self.turn = tf.placeholder("float", [None, 1])

        y0 = tf.reshape(self.state_input, [-1, 15, 15, 1])
        # 第一卷积层
        h1 = tf.nn.relu(self.conv2d(y0, W1) + b1)
        y1 = self.max_pool_3_3(h1)  # 5*5*16

        # 第二全连接层tf.concat([t1, t2], 0)
        h2 = tf.concat([tf.reshape(y1, [-1, 5 * 5 * 16]), self.turn], 1)
        self.Q_value = tf.matmul(h2, W2)+b2
        # 保存权重
        self.Q_weihgts = [W1, b1, W2, b2]

    def create_targetQ(self):
        # 网络权值
        W1 = self.weight_variable([5, 5, 1, 16])
        b1 = self.bias_variable([16])  # 5*5*16
        W2 = self.weight_variable([5*5*16+1, 225])
        b2 = self.bias_variable([1, 225])

        # 输入层
        # self.state_input = tf.placeholder("float", [None, self.state_dim])
        # self.turn = tf.placeholder("float", [None, 1])

        y0 = tf.reshape(self.state_input, [-1, 15, 15, 1])
        # 第一卷积层
        h1 = tf.nn.relu(self.conv2d(y0, W1) + b1)
        y1 = self.max_pool_3_3(h1)  # 5*5*16

        # 第二全连接层tf.concat([t1, t2], 0)
        h2 = tf.concat([tf.reshape(y1, [-1, 5 * 5 * 16]), self.turn], 1)
        self.targetQ_value = tf.matmul(h2, W2)+b2
        # 保存权重
        self.targetQ_weights = [W1, b1, W2, b2]

    def copy(self):
        """拷贝网络"""
        for i in range(len(self.Q_weihgts)):
            self.sess.run(
                tf.assign(self.targetQ_weights[i], self.Q_weihgts[i]))

    def train_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])

        Q_action = tf.reduce_sum(tf.multiply(
            self.Q_value, self.action_input), reduction_indices=1)  # mul->matmul
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.train = tf.train.AdamOptimizer(1e-3).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        """添加经验池"""
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append(
            [state, one_hot_action, reward, next_state, done])
        # 经验池满了
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        # 一个batch够了
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def modify_last_reward(self, new_reward):
        v = self.replay_buffer.pop()
        v[2] = new_reward
        self.replay_buffer.append(v)

    def train_Q_network(self):
        self.time_step += 1
        # 构建一个小的训练batch
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0][0] for data in minibatch]
        state_batch_turn = [data[0][1] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3][0] for data in minibatch]
        next_state_batch_turn = [data[3][1] for data in minibatch]
        # 构建训练数据
        y_batch = []

        """
            这里是计算新局面的估值                                                                                                                    Q'
            使用：targetQ 或 Q  网络来计算

            全局就在此处可能使用targetQ，作为对新状态的估值

            在csdn的代码中，使用targetQ的代码被注释掉了
        """
        # 计算Q'的所有值

        Q_value_batch = self.sess.run(self.Q_value, feed_dict={
            self.state_input: next_state_batch, self.turn: next_state_batch_turn})
        # Q_value_batch = self.sess.run(self.targetQ_value, feed_dict={
        #    self.state_input: next_state_batch[:, 0], self.turn: next_state_batch[:, 1]})

        # Q的值 Q = γ * max(Q')
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]  # 是否结束
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] +
                               GAMMA * np.max(Q_value_batch[i]))
        self.sess.run(self.train, feed_dict={
                      self.y_input: y_batch, self.action_input: action_batch, self.state_input: state_batch, self.turn: state_batch_turn})

        # 每一定轮次 拷贝Q网络到targetQ网络
        if self.time_step % self.targetQ_step == 0:
            self.epsilon *= 0.99  # 每次更新targetQ，减小随机选择落子点的概率
            self.copy()

    def egreedy_action(self, state):
        """含有随机 计算一步"""
        # 计算当前局面的所有Q值
        Q_value = self.sess.run(self.Q_value, feed_dict={
                                self.state_input: [state[0]], self.turn: [state[1]]})[0]
        min_v = Q_value[np.argmin(Q_value)] - 1  # 最小的Q_value -1
        valid_action = []

        for i in range(len(Q_value)):  # 遍历每一个落子点
            if state[0][i] == EMPTY:  # 空，可以落子
                valid_action.append(i)
            else:  # 有棋子，不可以落子
                Q_value[i] = min_v

        # 以 epsilon的概率随机落子
        if random.random() <= self.epsilon:
            l = len(valid_action)
            if l == 0:
                return -1
            else:
                return valid_action[random.randint(0, len(valid_action) - 1)]
        else:  # 其它清空，选取Q最大的点落子
            return np.argmax(Q_value)

    def action(self, state):
       # 计算当前局面的所有Q值
        Q_value = self.sess.run(self.Q_value, feed_dict={
            self.state_input: [state[0]], self.turn: [state[1]]})[0]

        min_v = Q_value[np.argmin(Q_value)] - 1  # 最小的Q_value -1
        valid_action = []

        for i in range(len(Q_value)):  # 遍历每一个落子点
            if state[0][i] == EMPTY:  # 空，可以落子
                valid_action.append(i)
            else:  # 有棋子，不可以落子
                Q_value[i] = min_v
        return np.argmax(Q_value)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, w):
        """定义卷积函数"""
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_3_3(self, x):
        """定义2*2最大池化层"""
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

    def save_model(self, save_path):
        saver = tf.train.Saver()
        path = saver.save(self.sess, 'mnist_model.ckpt')
        print(f"Model saved at {save_path}")

if __name__ == "__main__":
    # 一些超参数
    EPISODE = 10000
    """
        这个参数是每局最大的步数。
        在超过225的步数中,有一定概率是因为棋盘没满，但是发生了碰撞
        也有一定情况是棋盘已满。对这种情况，可以在action函数中添加判断
        语句，即在所有Q<0时，判别棋盘已满跳出；也可以不做处理，处理至
        300步满，第二种选择应该更好，因为它可以强化模型对每一个action的
        有子、无子的判断。

        通过对步数曲线的观测，还可以判断模型非法落子发生的概率情况

        最终选择，不做任何处理，保留棋盘满的情况。
    """
    STEP = 300
    TEST = 1

    chess = ChessBoard()  # 创建一个主棋盘

    agent = DQN()  # 创建网络对象
    agent.copy()  # 拷贝Q网络参数到targetQ

    for episode in range(EPISODE):
        # 初始一个棋盘
        chess.reset()
        state = chess.board
        camp = np.zeros([1])
        camp[0] = -1
        state = np.reshape(state, [-1])  # 二维数组转换为一维数组
        state = [state, camp]
        # 训练
        for step in range(STEP):
            # 自己下一步棋
            action_1d = agent.egreedy_action(state)  # 有随机概率的走一步
            action_2d = [math.floor(action_1d / ChessBoard.SIZE), action_1d %
                         ChessBoard.SIZE, camp]  # 转化为二维棋盘坐标
            
            # 在模拟棋盘上落子
            next_state_2d, reward, done, _ = chess.draw_XY(
                action_2d[0], action_2d[1])
            next_state = np.reshape(next_state_2d, [-1])
            # 构造数据
            if step % 2 == 0:
                camp[0] = 1
            else:
                camp[0] = -1
            next_state = [next_state, camp]
            # 定义奖励
            reward_agent = reward
            # 丢入经验池  执行训练
            agent.perceive(state, action_1d, reward, next_state, done)
            state = next_state
            # 判断这一步走了每    没走说明这一步是非法的，发生了碰撞，立即结束当前棋局，重开一局
            # 上面的判断不是很对
            # 成功落子后反而跳出了
            if done:
                chess.printChess()
                print("done step:%d  episode:%d" % (step, episode))
                agent.save_model("C:\\Users\\alienware\\Desktop\\AI-Homework-master")
                break

        """单步落子的查看，暂不跑
        if episode % 100 == 2:
            print("\n\nTest block")
            total_reward = 0
            # 测试 TEST盘棋
            for i in range(TEST):
                # 初始一个棋盘
                chess.reset()
                state = chess.board
                camp = np.zeros([1])
                camp[0] = -1
                state = np.reshape(state, [-1])
                state = [state, camp]
                for j in range(STEP):
                    # 开始走棋
                    action = agent.action(state)  # 按照Q网络的走一步
                    action = [math.floor(action/ChessBoard.SIZE), action %
                              ChessBoard.SIZE, camp]  # 转化为二维棋盘坐标

                    state, reward, done, _ = chess.draw_XY(
                        action[0], action[1])
                    state = np.reshape(state, [-1])

                    if j % 2 == 0:
                        camp[0] = 1
                    else:
                        camp[0] = -1
                    state = [state, camp]

                    total_reward += reward

                    # 打印棋盘
                    os.system("cls")
                    chess.printChess()
                    if camp == 1:
                        print("\n BLACK %d   %d" % (action[0], action[1]))
                    else:
                        print("\n WHITE %d   %d" % (action[0], action[1]))
                    time.sleep(2)  # 睡眠延时

                    # 结束判断
                    if done:
                        print("done step:%d  episode:%d" % (step, episode))
                        # print('done')
                        time.sleep(5)
                        break

            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
        """
