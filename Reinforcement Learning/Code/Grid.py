import random
class Grid:
    ACTION_UP = 0
    ACTION_LEFT = 1
    ACTION_DOWN = 2
    ACTION_RIGHT = 3
    ACTION_NOP = 4

    ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_NOP]
    ACTIONS_NAMES = ['UP','LEFT','DOWN','RIGHT', 'NOP']
    
    MOVEMENTS = {
        ACTION_UP: (1, 0),
        ACTION_LEFT: (0, -1),
        ACTION_DOWN: (-1, 0),
        ACTION_RIGHT: (0, 1),
        ACTION_NOP: (0, 0)
    }
    
    num_actions = len(ACTIONS)
    
    def __init__(self, n, m, wrong_action_p=0.1, alea=False):
        self.n = n
        self.m = m
        self.wrong_action_p = wrong_action_p
        self.alea = alea
        self.generate_game()
        
    def _position_to_id(self, x, y):
        """Gives the position id (from 0 to n)"""
        return x + y * self.n
    
    def _id_to_position(self, id):
        """Réciproque de la fonction précédente"""
        return (id % self.n, id // self.n)
    
    def get_possible_actions(self):
        return self.ACTIONS

    def generate_game(self):
        cases = [(x, y) for x in range(self.n) for y in range(self.m)]
        # hole = random.choice(cases)
        # cases.remove(hole)
        start = random.choice(cases)
        cases.remove(start)
        end = random.choice(cases)
        cases.remove(end)
        block = random.choice(cases)
        cases.remove(block)
        
        self.position = start
        self.end = end
        # self.hole = hole
        self.block = block
        self.counter = 0
        
        if not self.alea:
            self.start = start
        return self._get_state()
    
    def reset(self):
        if not self.alea:
            self.position = self.start
            self.counter = 0
            return self._get_state()
        else:
            return self.generate_game() 
    
    def _get_grille(self, x, y):
        grille = [
            [0] * self.n for i in range(self.m)
        ]
        grille[x][y] = 1
        return grille
    
    def _get_state(self):
        if self.alea:
            return [self._get_grille(x, y) for (x, y) in
                    [self.position, self.end, self.block]]
        return self._position_to_id(*self.position)
   
    def move(self, action):
        """
        takes an action parameter
        :param action : the id of an action
        :return ((state_id, end, block), reward, is_final, actions)
        """
        self.counter += 1
        if action not in self.ACTIONS:
            raise Exception('Invalid action')
        
        choice = random.random()
        if choice < self.wrong_action_p :
            action = (action + 1) % 5
        elif choice < 2 * self.wrong_action_p:
            action = (action - 1) % 5
            
        d_x, d_y = self.MOVEMENTS[action]
        x, y = self.position
        new_x, new_y = x + d_x, y + d_y
        
        if self.block == (new_x, new_y):
            return self._get_state(), -1, False, self.ACTIONS
        elif self.end == (new_x, new_y):
            self.position = new_x, new_y
            return self._get_state(), 1, True, self.ACTIONS
        elif new_x >= self.n or new_y >= self.m or new_x < 0 or new_y < 0:
            return self._get_state(), -1, False, self.ACTIONS
        elif self.counter > 190:
            self.position = new_x, new_y
            return self._get_state(), -10, True, self.ACTIONS
        else:
            self.position = new_x, new_y
            return self._get_state(), 0, False, self.ACTIONS
        
    def print(self):
        str = ""
        for i in range(self.n - 1, -1, -1):
            for j in range(self.m):
                if (i, j) == self.position:
                    str += "x"
                elif (i, j) == self.block:
                    str += "¤"
                # elif (i, j) == self.hole:
                #     str += "o"
                elif (i, j) == self.end:
                    str += "@"
                else:
                    str += "."
            str += "\n"
        print(str)        