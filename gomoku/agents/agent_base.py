class Agent:
    def __init__(self, name):
        self.name = name

    def select_action(self, state):
        raise NotImplementedError
