import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from TwoLayerNet import TwoLayerNet

class Agent:

    def __init__(self):
        self.maxMemory = 100_000
        self.batchSize = 1000
        self.lr = 0.01
        self.n_games = 0
        self.epsilon = (80, 200) # randomness 
        self.memory =  deque(maxlen=self.maxMemory)
        self.shortMemory = deque(maxlen=2)
        self.game  = SnakeGameAI(speed=240, block_size=40, w=640, h=640)
        self.inputSize = 14
        self.model = TwoLayerNet(self.inputSize, 256, 3, weight_init_std=np.sqrt(1/self.inputSize))
        self.gamma = 0.9


    # note to get this to work going to have to use convolutional layers 
    def get_state(self):
        while len(self.shortMemory) < 2:
            state = []
            for x in range(0, self.game.w, self.game.block_size):
                state.append(1)
            for y in range(0, self.game.h, self.game.block_size):
                state.append(1)
                for x in range(0, self.game.w, self.game.block_size):
                    position  = Point(x, y)
                    if position in self.game.snake:
                        state.append(0.75)
                    elif position == self.game.food:
                        state.append(0.25)
                    else:
                        state.append(0)
                state.append(1)
            for x in range(0, self.game.w, self.game.block_size):
                state.append(1)
            self.shortMemory.append(state)
        state = []
        for x in range(0, self.game.w, self.game.block_size):
            state.append(1)
        for y in range(0, self.game.h, self.game.block_size):
            state.append(1)
            for x in range(0, self.game.w, self.game.block_size):
                position  = Point(x, y)
                if position in self.game.snake:
                    state.append(0.75)
                elif position == self.game.food:
                    state.append(0.25)
                else:
                    state.append(0)
            state.append(1)
        for x in range(0, self.game.w, self.game.block_size):
            state.append(1)
        self.shortMemory.append(state)
        state = np.concatenate((np.array(self.shortMemory[0]), np.array(self.shortMemory[1])))
  #      state = np.concatenate((state, np.array(self.shortMemory[2])))
  #      state = np.concatenate((state, np.array(self.shortMemory[3])))
  #      state = np.concatenate((state, np.array(self.shortMemory[4])))
        return np.array([state], dtype="float")

    def get_state2(self):
        head = self.game.snake[0]
        point_l = Point(head.x - self.game.block_size, head.y)
        point_2l = Point(head.x - (2*self.game.block_size), head.y)
        point_r = Point(head.x + self.game.block_size, head.y)
        point_2r = Point(head.x + (2*self.game.block_size), head.y)
        point_u = Point(head.x, head.y - self.game.block_size)
        point_2u = Point(head.x, head.y - (2*self.game.block_size))
        point_d = Point(head.x, head.y + self.game.block_size)
        point_2d = Point(head.x, head.y + (2*self.game.block_size))
        
        dir_l = self.game.direction == Direction.LEFT
        dir_r = self.game.direction == Direction.RIGHT
        dir_u = self.game.direction == Direction.UP
        dir_d = self.game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.game.is_collision(point_r)) or 
            (dir_l and self.game.is_collision(point_l)) or 
            (dir_u and self.game.is_collision(point_u)) or 
            (dir_d and self.game.is_collision(point_d)),

            (dir_r and self.game.is_collision(point_2r)) or 
            (dir_l and self.game.is_collision(point_2l)) or 
            (dir_u and self.game.is_collision(point_2u)) or 
            (dir_d and self.game.is_collision(point_2d)),

            # Danger right
            (dir_u and self.game.is_collision(point_r)) or 
            (dir_d and self.game.is_collision(point_l)) or 
            (dir_l and self.game.is_collision(point_u)) or 
            (dir_r and self.game.is_collision(point_d)),

            (dir_u and self.game.is_collision(point_2r)) or 
            (dir_d and self.game.is_collision(point_2l)) or 
            (dir_l and self.game.is_collision(point_2u)) or 
            (dir_r and self.game.is_collision(point_2d)),

            # Danger left
            (dir_d and self.game.is_collision(point_r)) or 
            (dir_u and self.game.is_collision(point_l)) or 
            (dir_r and self.game.is_collision(point_u)) or 
            (dir_l and self.game.is_collision(point_d)),

            (dir_d and self.game.is_collision(point_2r)) or 
            (dir_u and self.game.is_collision(point_2l)) or 
            (dir_r and self.game.is_collision(point_2u)) or 
            (dir_l and self.game.is_collision(point_2d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.game.food.x < self.game.head.x,  # food left
            self.game.food.x > self.game.head.x,  # food right
            self.game.food.y < self.game.head.y,  # food up
            self.game.food.y > self.game.head.y  # food down
            ]
        return np.array([state], dtype=int)

    def get_action(self, state):
        randomness = self.epsilon[0] - self.n_games
        final_move = [0,0,0]
        if random.randint(0, self.epsilon[1]) < randomness:
            move = random.randint(0, 2)
        else:
            prediction = self.model.predict(state)
            move = np.argmax(prediction)
        final_move[move] = 1    
        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > self.batchSize:
            mini_sample = random.sample(self.memory, self.batchSize) # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)

    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state, dtype="float")
        next_state = np.array(next_state, dtype="float")
        action = np.array(action, dtype="float")
        reward = np.array(reward, dtype="float")
        #(n, x)
        if state.shape[0] == 1:
            # force 1 dimesion array to 2 so batch based calculation still work
            state = np.array([state], dtype="float")
            next_state = np.array([next_state], dtype="float")
            action = np.array([action], dtype="float")
            reward = np.array([reward], dtype="float")
            done = (done,)
        pred = self.model.predict(state)
        target = pred.copy()
        for idx in range(len(done)):
            Q_new = reward[idx]
            # this code forces deaths to be instant so the snake realises death 
            if not done[idx]:
                Q_new = (reward[idx]) + self.gamma * np.max(self.model.predict(next_state[idx]))
            target[idx][np.argmax(action)] = Q_new
        # Use backpropagoation to obtain a gradient
        grad = self.model.gradient(state, target)
##        if self.n_games > 200:
##            print(pred, "pred")
##            print(target, "target")
##        grad_numerical = self.model.numerical_gradient(state, target)
##        # Calculate the average of the absolute errors of individual weights
##        for key in grad_numerical.keys():
##            diff = np.average(np.abs(grad[key]-grad_numerical[key]))
##            print(key + ":" + str(diff))
        # Update
##        print(pred, "pred")
####        print(target, "target")
        for key in ("W1", "b1", "W2", "b2"):
            self.model.params[key] -= self.lr * grad[key]
##        print(self.model.predict(state), "new pred")

    def train(self):
        score = 0
        record = 0

        while True:
            # get old state
            state_old = agent.get_state2()
            # get move
            move = agent.get_action(state_old)
            # perform move and get new state
            reward, done, score = self.game.play_step(move)
            state_new = agent.get_state2()
            agent.train_short_memory(state_old, move, reward, state_new, done)
            agent.remember(state_old, move, reward, state_new, done)

            # if snake died 
            if done:
                self.game.reset()
                self.n_games += 1
                self.train_long_memory()
                if score > record:
                    record = score
                print("Game:" + str(self.n_games) + "\nScore:" + str(score) + "\nRecord:" + str(record) + "\n\n")

if __name__ == "__main__":
    agent = Agent()
    agent.train()

