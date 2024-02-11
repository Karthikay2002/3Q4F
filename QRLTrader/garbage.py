def get_reward(self, a, y):
    r = 0.0
    if a == 1 and y == 1:
        if self.m == -1:
            r = 3.0
        elif self.m == 0:
            r = 2.0
        elif self.m == 1:
            r = 1.0
    elif a == -1 and y == -1:
        if self.m == 1:
            r = 3.0
        elif self.m == 0:
            r = 2.0
        elif self.m == -1:
            r = 1.0
    elif a == 0:
        if y != 0:
            r = -1.0
    elif a == 1 and y == -1:
        if self.m == 1:
            r = -3.0
        elif self.m == 0:
            r = -2.0
        else:
            r = -1.0
    elif a == -1 and y == 1:
        if self.m == -1:
            r = -3.0
        elif self.m == 0:
            r = -2.0
        elif self.m == 1:
            r = -1.0
    return r
