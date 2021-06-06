import numpy as np
import random



class NPUZZLE:
    def __init__(
            self,
            grid_x,
            grid_y,
            random_extent=0
    ):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.total_num = self.grid_x * self.grid_y

        self.targ_mat = np.arange(1, self.total_num+1).reshape((self.grid_x, self.grid_y))
        self.targ_mat[grid_x-1,grid_y-1] = 0
        self.cur_mat = self.targ_mat.copy()
        self.org_mat = self.cur_mat.copy()
        # if random_extent:
        #     self.randomize(random_extent*self.grid_x**2)
        # else:
        #     self.randomize(2*self.grid_x**2)  
        self.blank_loc = None

        self.punish = False
        self.done = False
        self.time_costs = 0
        self.last_move = 0

        

    def reset(self, init_mat):
        # self.cur_mat = np.arange(self.total_num)
        # np.random.shuffle(self.cur_mat)
        # self.cur_mat = np.reshape(self.cur_mat, (self.grid_x, self.grid_y))
        self.cur_mat = init_mat

        return self.get_state()

    def find_blank(self):
        blank_loc = np.where(self.cur_mat==0)
        return np.hstack((blank_loc[0], blank_loc[1]))

    def get_feasible_action(self, blank_loc):
        # four corners
        blank_loc = self.find_blank()
        if blank_loc[0] == 0 and blank_loc[1] == 0:
            ''' top left corner, swap with right and bottom '''
            feasible_action = [1, 3]
        elif blank_loc[0] == self.grid_x-1 and blank_loc[1] == 0:
            ''' bottom left corner, swap with right and top '''
            feasible_action = [1, 2]
        elif blank_loc[0] == 0 and blank_loc[1] == self.grid_y-1:
            ''' top right corner, swap with left and bottom '''
            feasible_action = [0, 3]
        elif blank_loc[0] == self.grid_x-1 and blank_loc[1] == self.grid_y-1:
            ''' bottom right corner, swap with left and top '''
            feasible_action = [0, 2]

        # four edges
        elif blank_loc[0] == 0:
            ''' top edge, swap with left, right and bottom '''
            feasible_action = [0, 1, 3]
        elif blank_loc[0] == self.grid_x - 1:
            ''' bottom edge, swap with left, right and top '''
            feasible_action = [0, 1, 2]
        elif blank_loc[1] == 0:
            ''' left edge, swap with right, top and bottom '''
            feasible_action = [1, 2, 3]
        elif blank_loc[1] == self.grid_y - 1:
            ''' right edge, swap with left, top and bottom '''
            feasible_action = [0, 2, 3]
        else:
            feasible_action = [0, 1, 2, 3]

        return feasible_action

    def swap(self, action):
        temp = self.cur_mat[self.blank_loc[0], self.blank_loc[1]]
        if action == 0:
            ''' swap with left '''
            self.cur_mat[self.blank_loc[0], self.blank_loc[1]] = self.cur_mat[self.blank_loc[0], self.blank_loc[1] - 1]
            self.cur_mat[self.blank_loc[0], self.blank_loc[1] - 1] = temp
        elif action == 1:
            ''' swap with right '''
            self.cur_mat[self.blank_loc[0], self.blank_loc[1]] = self.cur_mat[self.blank_loc[0], self.blank_loc[1] + 1]
            self.cur_mat[self.blank_loc[0], self.blank_loc[1] + 1] = temp
        elif action == 2:
            ''' swap with top '''
            self.cur_mat[self.blank_loc[0], self.blank_loc[1]] = self.cur_mat[self.blank_loc[0] - 1, self.blank_loc[1]]
            self.cur_mat[self.blank_loc[0] - 1, self.blank_loc[1]] = temp
        else:
            ''' swap with bottom '''
            self.cur_mat[self.blank_loc[0], self.blank_loc[1]] = self.cur_mat[self.blank_loc[0] + 1, self.blank_loc[1]]
            self.cur_mat[self.blank_loc[0] + 1, self.blank_loc[1]] = temp
        self.blank_loc = self.find_blank()
        self.last_move = action
        # except:
        #     print('error x is {} error y is {}'.format(self.blank_loc[0], self.blank_loc[1]))

    def lock(self):
        pass

    def get_state(self):
        state = np.reshape(self.cur_mat, -1)

        return state

    def get_distance(self):
        ''' Euclidean Distance '''
        ''' Manhattan Distance '''
        mht_dis = 0
        for i in range(self.total_num):
            org_ploc = np.where(self.targ_mat == i)
            cur_ploc = np.where(self.cur_mat == i)

            org_vector = np.stack((org_ploc[0], org_ploc[1]))
            cur_vector = np.stack((cur_ploc[0], cur_ploc[1]))

            mht_dis += np.linalg.norm(cur_vector - org_vector, ord=1)

        reward = mht_dis

        return reward

    def step(self, action):
        ''' get the feasible action list'''
        self.blank_loc = self.find_blank()
        self.feasible_action = self.get_feasible_action(list(self.blank_loc))

        try:
            self.swap(action)
            self.punish = False
        except:
            ava_action = np.random.randint(0, len(self.feasible_action), 1)[0]
            ava_action = self.feasible_action[ava_action]
            self.swap(ava_action)
            self.punish = True

        if self.punish == True:
            reward = - 50
        else:
            reward = - self.get_distance()

        state = self.get_state()

        done = self.is_done()

        self.time_costs = 0

        return state, reward, done

    def is_done(self):
        if (self.cur_mat == self.targ_mat).all():
            self.done = 1
        else:
            self.done = 0

        return self.done

    def randomize(self, step_num=5):
        if (self.cur_mat != self.targ_mat).all():
            print('Not beginning, cannot be randomzed!')
            return
        else:
            self.blank_loc = self.find_blank() 
            for i in range(step_num):
                feasible_action = self.get_feasible_action(self.blank_loc)
                action = random.choice(feasible_action)
                self.step(action)
            self.org_mat = self.cur_mat.copy()



if __name__ == "__main__":
    agent = NPUZZLE(4,4)
