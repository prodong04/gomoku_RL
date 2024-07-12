import numpy as np

class GomokuEnv:
    def __init__(self, board_size=19):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        return self.board

    def step(self, action):
        x, y = action
        if self.board[x, y] != 0:
            return self.board, -1, True, {"invalid": True}
        
        self.board[x, y] = self.current_player
        reward, done = self.check_winner(x, y)
        self.current_player = 3 - self.current_player
        return self.board, reward, done, {}

    def check_winner(self, x, y):
        # 육목 승리 조건 체크 구현
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for direction in [1, -1]:
                nx, ny = x + direction*dx, y + direction*dy
                while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == self.current_player:
                    count += 1
                    nx += direction*dx
                    ny += direction*dy
                if count >= 6:
                    return 1, True
        return 0, np.all(self.board != 0)

    def render(self):
        print(self.board)

env = GomokuEnv()
env.reset()
env.render()
