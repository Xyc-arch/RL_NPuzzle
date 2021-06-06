import numpy as np
import random
from simulator import NPUZZLE
import time
import math

# class AstarNode:
#     def __init__(self,parent,state,degree=0,h=0,f=0):
#         self.parent=parent
#         self.state=state
#         self.degree=degree
#         self.h = h 
#         self.f = f


class ASTARAgent:
    def __init__(self,simulator,MaxDegree=100):
        self.simulator = simulator
        self.open=[self.createNode(None, self.simulator.cur_mat)]
        self.close=[self.createNode(None, self.simulator.cur_mat)]
        self.MaxDegree= math.floor(1.2 * self.simulator.grid_x**2)  #深度限制，到达此深度未找到解便返回
        self.MaxLoop=100*self.simulator.grid_x**2
        self.solved = False
        self.loop_num = 0
        self.nodeNum = 0


    def createNode(self, parent, state, degree=0, h=0, f=0):
        class AstarNode:
            def __init__(self,parent,state,degree,h,f):
                self.parent=parent
                self.state=state
                self.degree=degree
                self.h = h 
                self.f = f

        return AstarNode(parent, state, degree, h, f)

    #判断是否有解
    def hasSolve(self):
        pass
    #获取逆序数
    def getreVersNum(self,state):
        pass


    def isInTable(self,node,table):
        for i in table:
            if (i.state==node.state).all() and i.degree==node.degree:
                return True
        return False

    def orderOpen(self):
        fMin = self.open[-1].f
        for i in range(len(self.open)):
            if self.open[i].f <= fMin:
                fMin = self.open[i].f
                self.open = self.open[:i] + self.open[i+1:] + [self.open[i]]

    def reAction(self, action):
        if action == 0:
            self.simulator.step(1)
        elif action == 1:
            self.simulator.step(0)
        elif action == 2:
            self.simulator.step(3)
        else:
            self.simulator.step(2)

    def showLine(self, show_tag = False):
        endState=self.open[-1]
        road=[endState]
        while(True):
            if(endState.parent):
                endState=endState.parent
                road.append(endState)
            else:
                break
        road.reverse()
        step = 0
        for j in road:
            step += 1
            if show_tag:
                print(j.state)
                print('->')
        if show_tag:
            print('Step number is {}'.format(step-1))
            print('Target is:')
            print(self.simulator.targ_mat)
            print('Initial is:')
            print(self.simulator.org_mat)
            if self.solved:
                print('Successfully solved in {} loops'.format(self.loop_num))
            else:
                print('Not solved!')
        return step
        

    def search(self):
        simulator = self.simulator
        while(True):
            self.loop_num += 1
            if self.loop_num > self.MaxLoop:
                return
            else:
                pass
            if self.loop_num % 500 == 0:
                print('Loop number achieve {}'.format(self.loop_num))
            else:
                pass
            if(len(self.open)):
                extandState=self.open[-1]
                if extandState.degree > self.MaxDegree:
                    return
                else:
                    pass
                simulator.cur_mat = extandState.state.copy()
                #print('current = ')  ###
                #print(simulator.cur_mat)
                #print('-'*30) ###
                flag=False
                feasible_action = simulator.get_feasible_action(simulator.blank_loc)
                for action in [0,1,2,3]:
                    if action not in feasible_action:
                        continue
                    else:
                        simulator.step(action)
                        state = simulator.cur_mat.copy()
                        #print(state) ##
                        #print('*'*30) ##
                        nodeState = self.createNode(extandState,state,extandState.degree+1)
                        self.nodeNum += 1 # we count the node number here
                        target = simulator.targ_mat
                        # nodeState.h = np.linalg.norm(np.reshape((state-target),-1),ord=1)
                        nodeState.h = simulator.get_distance()
                        nodeState.f = nodeState.h + nodeState.degree
                        if (state == target).all():
                            self.open.append(nodeState)
                            self.solved = True
                            return True
                        elif ((not self.isInTable(nodeState,self.close)) and (not self.isInTable(nodeState,self.open))):
                            self.open.append(nodeState)
                            flag = True
                            self.reAction(action)
                        else:
                            self.reAction(action)
                            continue
                if(not flag):
                    self.open.pop()
                else:
                    self.close.append(extandState)
                    self.open.remove(extandState)
                    self.orderOpen()

            else:
                return False


# start = time.clock()
# end = time.clock()
# print(end-start)


def single_experiment(times = 50, size = 3, rand = 2):

    # criteria part:
    whole_time = 0
    whole_step = 0
    whole_node = 0
    whole_success = 0
    round_num = 0

    for i in range(times):
        agent = ASTARAgent(NPUZZLE(size,size, rand))
        start = time.clock()
        agent.search()
        end = time.clock()
        if agent.solved:
            whole_time += end - start
            whole_step += agent.showLine()
            whole_node += agent.nodeNum
            whole_success += 1
            round_num += 1
            print('Round {}'.format(round_num))
    
    success_rate = whole_success/times
    average_time = whole_time/whole_success
    average_step = whole_step/whole_success
    average_node = whole_node/whole_success
    print('This is a A* experiment')
    print('Success rate is {} (when the max dgree is {} (when the max loop is {} (when the randomization is {}'.format(success_rate, agent.MaxDegree, agent.MaxLoop, rand*size**2))
    print('Average time cost is {}'.format(average_time))
    print('Average step of solution is {}'.format(average_step))
    print('Average node number is {}'.format(average_node))
    print(40*'-')


def experiment(size=3, repeat=15):
    for rand in range(1,5):
        single_experiment(repeat, size, rand)

# single_experiment(50)

experiment()




