import random
from algori_func import *
import multiprocessing
import pandas as pd


# 这是参数
class para():
    J = 7
    Es = 6
    S = 7
    W = J*S
    # JmNumber = 8
    JmNumber = 9 #所有不需要机器的操作看作是一个虚拟机器
    fmax = 0.8
    nObj = 3
    T = 15
    f_max = 0.7
    f_1 = 0.3
    f_2 = 0.7
    pop_size = 60
    terminate = 60
    XOVR = 0.8
    MUTR = 0.4
    Hf = 0.5
    gap = 10
    # archive = 30  # 存档数目
    u = np.array([0.0099, 0.0098, 0.0085, 0.0083, 0.0088, 0.0098])  #恢复率
    lamda = np.array([0.0094, 0.0098, 0.0088, 0.0083, 0.0099,0.01])    #疲劳率
    # m_cost = np.array([10,10,10,20,20,20,20,10,5])
    m_cost = np.array([0.21, 0.22, 0.21, 0.42, 0.41,0.42 , 0.39, 0.19, 0])
    w_cost = np.array([0.60,0.21,0.22,0.19,0.61,0.21])
    # alpha = [3.4,1.1]
    # 原模型，7个工件，那我就每个工件都先*1，再换规模
    # TM = np.dot(10,np.array([[[7, 3, 4, 3, 9, 9, 4], [6, 6, 5, 4, 6, 5, 5], [15, 13, 14, 14, 13, 18, 13]],
    #        [[3, 5, 7, 6, 5, 5, 9], [6, 6, 5, 4, 5, 5, 4], [11, 15, 17, 16, 15, 12, 10]],
    #        [[9, 3, 3, 5, 5, 7, 9], [4, 5, 5, 5, 6, 5, 6], [13, 16, 13, 17, 8, 12, 15]],
    #        [[3, 8, 9, 9, 6, 6, 8], [4, 5, 6, 4, 6, 6, 5], [12, 10, 15, 16, 14, 16, 8]],
    #        [[9, 8, 3, 5, 5, 7, 7], [4, 5, 5, 5, 4, 4, 6], [15, 11, 14, 18, 14, 18, 18]],
    #        [[6, 5, 5, 8, 5, 4, 6], [5, 4, 6, 6, 6, 4, 4], [8, 10, 13, 16, 12, 8, 12]],
    #        [[7, 9, 3, 5, 6, 3, 4], [6, 6, 4, 5, 5, 6, 5], [11, 8, 13, 18, 16, 14, 17]],
    #        [[7, 9, 3, 5, 6, 3, 4], [6, 5, 6, 6, 4, 6, 4], [10, 15, 17, 17, 12, 16, 15]],
    #        [[6, 9, 9, 3, 5, 4, 9], [5, 6, 5, 6, 5, 5, 4], [15, 18, 10, 15, 16, 15, 11]],
    #        [[7, 9, 6, 8, 7, 8, 3], [5, 4, 4, 4, 5, 6, 5], [16, 13, 10, 14, 11, 13, 9]],
    #        [[9, 3, 9, 5, 7, 6, 4], [6, 6, 5, 5, 6, 6, 4], [9, 9, 18, 11, 18, 13, 12]],
    #        [[4, 8, 3, 4, 5, 3, 9], [5, 4, 5, 4, 5, 6, 4], [10, 11, 9, 14, 15, 14, 13]]]))#所有数乘以10
    # 每个员工对于加工某个工件的偏好度(员工-工件) # 每个工件属于不同的工件种类
    #每个工件可以被哪些机器操作，工件-工序-机器
    # 工件-工序-机器
    JM = np.array([[[1,2,3],[1,2,3],[9],[9],[1,2,3],[1,2,3],[1,2,3]],
                   [[8],[1,2,3],[1,2,3],[1,2,3],[8],[9],[4,5]],
                  [[4,5],[8],[6,7],[8],[1,2,3],[6,7],[9]],
                   [[9],[9],[8],[1,2,3],[6,7],[8],[9]],
                   [[6,7],[1,2,3],[9],[6,7],[9],[1,2,3],[1,2,3]],
                    [[9],[6,7],[4,5],[1,2,3],[6,7],[6,7],[8]],
                     [[9],[6,7],[6,7],[6,7],[4,5],[4,5],[6,7]]])

    # 工件-工序-对应机器的时间
    # TM = np.array([[[6,3,3],[5,3,4],[3],[4],[5,5,4],[6,7,3],[6,5,5]],
    #               [[1],[6,5,4],[6,6,4],[5,5,7],[5],[4],[2,4]],
    #               [[3,2],[3],[3,3],[2],[4,4,4],[6,5],[6]],
    #               [[2],[3],[2],[8,8,7],[5,3],[4],[3]],
    #                [[3,2],[6,4,4],[4],[6,7],[5],[5,6,8],[8,8,8]],
    #                [[9],[3,4],[2,3],[7,5,9],[6,7],[3,5],[7]],
    #                [[0],[9,8],[8,9],[4,6],[7,8],[9,7],[4,4]]])
    TM = np.array([[[60, 30, 30], [50, 30, 40], [30], [40], [50, 50, 40], [60, 70, 30], [60, 50, 50]],
                   [[10], [60, 50, 40], [60, 60, 40], [50, 50, 70], [50], [40], [20, 40]],
                   [[30, 20], [30], [30, 30], [20], [40, 40, 40], [60, 50], [60]],
                   [[20], [30], [20], [80, 80, 70], [50, 30], [40], [30]],
                   [[30, 20], [60, 40, 40], [40], [60, 70], [50], [50, 60, 80], [80, 80, 80]],
                   [[90], [30, 40], [20, 30], [70, 50, 90], [60, 70], [3, 50], [70]],
                   [[0], [90, 80], [80, 90], [40, 60], [70, 80], [90, 70], [40, 40]]])
    # 每个机器可以被哪些人员操控,最后一个虚拟工序所有人都可以做,机器-人员
    JW = np.array([[1, 3, 4, 5], [1, 3, 4, 5], [1, 3, 4, 5], [1, 5], [1, 5], [1, 5], [1, 5], [1, 2, 5, 6], [1, 2, 3, 4, 5, 6]])
    # 员工对设备的操作能力与偏好（与培训时间、熟练）有关，还有是否是机加工有关，机器位置、环境、上下料操作的难易
    # 操作、设备都不同，人因的偏好不同
    # （机器-人偏好）（影响恢复率和疲劳率）偏好越大，恢复越快
    Pr = np.array([[1.05, 1.08, 1.13, 1.17], [1.05, 1.08, 1.13, 1.15], [1.08, 1.03,  1.15, 1.13],
                   [1.22, 1.23], [1.24, 1.22], [1.26, 1.22], [1.25, 1.25], [1.05, 1.05,1.18,1.11],
                   [0.85, 0.78, 0.93, 0.88, 0.82, 0.85]])
    # Pr = np.array([[0.85, 0.78, 0.93],[0.6, 0.82, 0.85],[ 0.95, 0.85, 0.78],
    #                [0.93, 0.6], [0.94,0.82],[0.6, 0.82], [0.85, 0.95], [0.85, 0.95],
    #                [1.05, 1.08, 1.13,  1.05, 1.08, 1.13]])
    # 人换机器需要的时间（考虑路程和适应时间)：机器-机器
    # Trans_t = np.array([[0, 2, 1, 2, 2, 1, 3, 2, 1],
    #                     [2, 0, 3, 2, 1, 3, 2, 1, 2],
    #                     [1, 3, 0, 1, 3, 2, 3, 1, 1],
    #                     [2, 2, 1, 0, 1, 3, 2, 3, 1],
    #                     [2, 1, 3, 1, 0, 2, 1, 1, 1],
    #                     [1, 3, 2, 3, 2, 0, 2, 2, 2],
    #                     [3, 2, 3, 2, 1, 2, 0, 3, 1],
    #                     [2, 1, 1, 3, 1, 2, 3, 0, 1],
    #                     [1, 2, 1, 1, 1, 2, 1, 1, 0]])
    # 人员的移动时间
    Trans_t = np.array([[0, 12, 11, 12, 12, 11, 13, 12, 11],
                        [12, 0, 13, 12, 11, 13, 12, 11, 12],
                        [11, 13, 0, 11, 13, 12, 13, 11, 11],
                        [12, 12, 11, 0, 11, 13, 12, 13, 11],
                        [12, 12, 13, 11, 0, 12, 11, 11, 11],
                        [11, 13, 12, 13, 12, 0, 12, 12, 12],
                        [13, 12, 13, 12, 11, 12, 0, 13, 11],
                        [12, 11, 11, 13, 11, 12, 13, 0, 11],
                        [11, 12, 11, 11, 11, 12, 11, 11, 0]])
    # Trans_t = np.array([[0, 3, 4, 3, 9, 9, 4, 3, 2],
    #                     [3, 0, 5, 4, 6, 5, 5, 5, 2],
    #                     [4, 5, 0, 4, 3, 8, 3, 4, 2],
    #                     [3, 4, 4, 0, 8, 7, 9, 3, 2],
    #                     [9, 6, 3, 8, 0, 4, 6, 3, 2],
    #                     [9, 5, 8, 7, 4, 0, 5, 3, 4],
    #                     [4, 5, 3, 9, 6, 5, 0, 6, 3],
    #                     [3, 5, 4, 3, 3, 3, 6, 0, 3],
    #                     [2, 2, 2, 2, 2, 4, 3, 3, 0]])

parse = para()

# 算法函数
def NSGA2():
    pop = initialize()
    targ_pop = target_pop(pop)
    Zmin = np.array(np.min(targ_pop, 0)).reshape(1, parse.nObj)  # 求理想点
    ranks = nonDominationSort(targ_pop)
    Pop_MOP = pop[ranks == 0]
    EPs = copy.deepcopy([list(Pop_MOP[i]) for i in range(len(Pop_MOP))])
    iter = 0
    while iter < parse.terminate:
        print("第{name}次迭代".format(name=iter))
        matingpool = random.sample(range(parse.pop_size), parse.pop_size)  # 就是把1-popsize-1打乱顺序
        ChrPops = copy.deepcopy(pop)
        # 交叉
        N = pop.shape[0]
        pop = pop[matingpool, :]
        Chrom1 = pop[:int(N / 2)]
        Chrom2 = pop[int(N / 2):]
        for k in range(N):
            s1 = Chrom1[np.random.choice(int(N / 2), 1)][0]
            s2 = Chrom2[np.random.choice(int(N / 2), 1)][0]
            ChrPops[k] = cross_mutation(s1,s2)
            target_ChrPops = target(ChrPops[k])

            ##更新外部存档
            ep = True  # 决定y是否放进EPs,True就放
            delete = []  # 装EPs个体
            for EP in EPs:  # EPs是所有支配个体，EPs外部存档
                fun_EP = target(EP)
                if isDominates(fun_EP, target_ChrPops):  # EPs[k]支配y
                    ep = False
                    break
                elif isDominates(target_ChrPops, fun_EP):  # 存在y支配EPs[k]
                    delete.append(EP)  # 准备删除EPs[k]
            if ep:  # 存在y支配EPs[k]或者没有任何支配关系
                EPs.append(list(ChrPops[k]))
                for delete_i in delete[::-1]:
                    EPs.remove(delete_i)

        chrTarg = target_pop(ChrPops)
        pop, targ_pop = optSelect(pop, targ_pop, ChrPops, chrTarg)

        # mixpop = np.concatenate((pop, ChrPops), axis=0)
        # mixpopfun = target_pop(mixpop)
        # Zmin = np.array(np.min(mixpopfun, 0)).reshape(1, parse.nObj)  # 更新理想点
        iter += 1

    # IGD
    Zmin = np.array(np.min(targ_pop, 0)).reshape(1, parse.nObj)  # 求理想点
    Zmax = np.array(np.max(targ_pop, 0)).reshape(1, parse.nObj)  # 求负理想点
    targ_pop = target_pop(pop)
    Target_Ep = target_pop(EPs)
    score_GD = GD(targ_pop, Target_Ep, Zmin, Zmax)
    score_IGD = IGD(targ_pop, Target_Ep,Zmin, Zmax)
    score_HV = HV(Target_Ep,Zmin, Zmax)
    return pop,targ_pop, score_GD, score_IGD, score_HV

def MOEAD():
    lamda , pop_size = uniformpoint(parse.pop_size, parse.nObj)  #  Z是向量,N是向量个数(一般小于POP_SIZE)
    pop = initialize() # 初始化种群
    targ_pop = target_pop(pop)
    Zmin = np.array(np.min(targ_pop, 0)).reshape(1, parse.nObj)  # 理想点（M个目标值 ）[1,1,1]三个目标的群体最小值
    B = look_neighbor(lamda)

    # 迭代过程
    ranks = nonDominationSort(targ_pop)
    # Target_MOP = targ_pop[ranks == 0]
    Pop_MOP = pop[ranks==0]
    EPs = copy.deepcopy([list(Pop_MOP[i]) for i in range(len(Pop_MOP))])
    for gen in range(parse.terminate):
        print("第{name}次迭代".format(name=gen))
        for i in range(pop_size):
            ##基因重组，从B(i)中随机选取两个序列k，l
            k = random.randint(0, parse.T - 1)
            l = random.randint(0, parse.T - 1)
            # y = cross_mutation(pop[B[i][k]], pop[B[i][l]])
            y = crossover(pop[B[i][k]], pop[B[i][l]])
            y = mutation(y)
            t_y = target(y)

            ##更新z
            for j in range(len(Zmin[0])):
                if t_y[j] < Zmin[0][j]:
                    Zmin[0][j] = t_y[j]
            ##更新领域解
            for j in range(len(B[i])):
                gte_xi = Tchebycheff(pop[B[i][j]], lamda[B[i][j]], Zmin[0])
                gte_y = Tchebycheff(y, lamda[B[i][j]], Zmin[0])
                if (gte_y <= gte_xi):
                    pop[B[i][j]] = y

            ##更新外部存档
            ep = True  # 决定y是否放进EPs,True就放
            delete = []  # 装EPs个体
            for EP in EPs:  # EPs是所有支配个体，EPs外部存档
                fun_EP = target(EP)
                if isDominates(fun_EP, t_y) or fun_EP.all()==t_y.all():  # EPs[k]支配y
                    ep = False
                    break
                elif isDominates(t_y, fun_EP):  # 存在y支配EPs[k]
                    delete.append(EP)  # 准备删除EPs[k]
            if ep: #存在y支配EPs[k]或者没有任何支配关系
                EPs.append(list(y))
                for delete_i in delete[::-1]:
                    EPs.remove(delete_i)

        if gen % 10 == 0:
            print("%d gen has completed!\n" % gen)

    Zmin = np.array(np.min(targ_pop, 0)).reshape(1, parse.nObj)  # 求理想点
    Zmax = np.array(np.max(targ_pop, 0)).reshape(1, parse.nObj)  # 求负理想点
    targ_pop = target_pop(pop)
    Target_Ep = target_pop(EPs)
    score_GD = GD(targ_pop, Target_Ep, Zmin, Zmax)
    score_IGD = IGD(targ_pop, Target_Ep, Zmin, Zmax)
    score_HV = HV(Target_Ep, Zmin, Zmax)
    return pop,targ_pop, score_GD, score_IGD, score_HV

def SPEA2():
    pop = initialize()
    archieve = initialize()
    mixpop = np.concatenate((pop, archieve), axis=0)
    F,D = spea_cal_fitness(mixpop)
    archieve = update(F,D,mixpop,archieve)
    for i in range(parse.terminate):
        mix = np.concatenate((pop, archieve), axis=0)
        pop = selection(mix)
        N = pop.shape[0]
        Chrom1 = pop[:int(N / 2)]
        Chrom2 = pop[int(N / 2):]
        ChrPops = copy.deepcopy(pop)
        for k in range(N):
            s1 = Chrom1[np.random.choice(int(N / 2), 1)][0]
            s2 = Chrom2[np.random.choice(int(N / 2), 1)][0]
            ChrPops[k] = crossover(s1, s2)
            ChrPops[k] = mutation(ChrPops[k])
            # target_ChrPops = target(ChrPops[k])
        mixpop = np.concatenate((pop, archieve), axis=0)
        F, D = spea_cal_fitness(mixpop)
        archieve = update(F, D, mixpop,archieve)
    # IGD
    targ_pop = target_pop(pop)
    ranks = nonDominationSort(targ_pop)
    # Target_MOP = targ_pop[ranks == 0]
    Pop_MOP = archieve[ranks == 0]
    Target_Ep = target_pop(Pop_MOP)
    Zmin = np.array(np.min(targ_pop, 0)).reshape(1, parse.nObj)  # 求理想点
    Zmax = np.array(np.max(targ_pop, 0)).reshape(1, parse.nObj)  # 求负理想点
    score_GD = GD(targ_pop, Target_Ep, Zmin, Zmax)
    score_IGD = IGD(targ_pop, Target_Ep, Zmin, Zmax)
    score_HV = HV(Target_Ep, Zmin, Zmax)
    return pop,targ_pop, score_GD, score_IGD, score_HV

def IMOEAD():
    lamda , pop_size = uniformpoint(parse.pop_size, parse.nObj)  #  Z是向量,N是向量个数(一般小于POP_SIZE)
    pop = initialize() # 初始化种群
    targ_pop = target_pop(pop)
    Zmin = np.array(np.min(targ_pop, 0)).reshape(1, parse.nObj)  # 理想点（M个目标值 ）[1,1,1]三个目标的群体最小值
    B = look_neighbor(lamda)

    # 迭代过程
    ranks = nonDominationSort(targ_pop)
    Pop_MOP = pop[ranks==0]
    EPs = copy.deepcopy([list(Pop_MOP[i]) for i in range(len(Pop_MOP))])
    for gen in range(parse.terminate):
        print("第{name}次迭代".format(name=gen))
        # ChrPops = copy.deepcopy(pop)

        ## 全局搜索TOPSIS
        for i in range(pop_size):
            targ_pop = target_pop(pop)
            score = TOPSIS(targ_pop[B[i]], lamda[i])
            k_1 = roulette((-1) * score * 10 + 10)  # score越小越好
            k_2 = roulette((-1) * score * 10 + 10)
            if k_1 == None or k_2 == None:
                k_1 = random.randint(0, parse.T - 1)
                k_2 = random.randint(0, parse.T - 1)
            y = crossover(pop[B[i][k_1]], pop[B[i][k_2]])
            y = mutation(y)
            t_y = target(y)

            ##更新领域解
            for j in range(len(B[i])):
                gte_xi = penalty_based_boundary(pop[B[i][j]], lamda[B[i][j]], Zmin[0])
                gte_y = penalty_based_boundary(y, lamda[B[i][j]], Zmin[0])
                T = parse.terminate
                if (gte_y <= gte_xi):
                    pop[B[i][j]] = y
                elif random.random() < math.exp(-(gte_y - gte_xi)*gen/T):
                    # print('gte_y - gte_xi',gte_y - gte_xi)
                    pop[B[i][j]] = y
            ##更新z
            for j in range(len(Zmin[0])):
                if t_y[j] < Zmin[0][j]:
                    Zmin[0][j] = t_y[j]
            ##更新外部存档
            ep = True  # 决定y是否放进EPs,True就放
            delete = []  # 装EPs个体
            for EP in EPs:  # EPs是所有支配个体，EPs外部存档
                fun_EP = target(EP)
                if isDominates(fun_EP, t_y):  # EPs[k]支配y
                    ep = False
                    break
                elif isDominates(t_y, fun_EP):  # 存在y支配EPs[k]
                    delete.append(EP)  # 准备删除EPs[k]
            if ep:  # 存在y支配EPs[k]或者没有任何支配关系
                EPs.append(list(y))
                for delete_i in delete[::-1]:
                    EPs.remove(delete_i)


        ## 扰动算子产生新解
        LS = local_search()
        LS_pool = [LS.LS1, LS.LS2, LS.LS3, LS.LS4]
        LS_weight = [1, 1, 1, 1]

        targ_pop = target_pop(pop)
        ranks = nonDominationSort(targ_pop)
        ChrPops = pop[ranks == 0]
        newPop = []
        # newPop = np.zeros((len(ChrPops),len(pop[0])))
        Zmin = np.array(np.min(targ_pop, 0)).reshape(1, parse.nObj)  # 求理想点
        if gen < parse.terminate / 2:
            for i in range(len(ChrPops)):
                for LS_choice in LS_pool:
                    flag = False
                    y_min = LS_choice(ChrPops[i])
                    t_y_min = target(y)
                    for k in range(10): #邻域
                        y = LS_choice(ChrPops[i])
                        t_y = target(y)
                        if t_y[0]<t_y_min[0]:
                            y_min = y
                            t_y_min = t_y
                    for j in range(len(Zmin[0])):
                        if t_y_min[j] < Zmin[0][j]:
                            flag = True
                # LS_choice = random.choice(LS_pool)
                if flag: #有能更新领域解就可以加0.1
                    LS_weight[LS_weight == LS_choice] += 0.1
                newPop.append(list(y_min))
                # ##更新外部存档
                # ep = True  # 决定y是否放进EPs,True就放
                # delete = []  # 装EPs个体
                # for EP in EPs:  # EPs是所有支配个体，EPs外部存档
                #     fun_EP = target(EP)
                #     if isDominates(fun_EP, t_y) or fun_EP.all() == t_y.all():  # EPs[k]支配y
                #         ep = False
                #         break
                #     elif isDominates(t_y, fun_EP):  # 存在y支配EPs[k]
                #         delete.append(EP)  # 准备删除EPs[k]
                # if ep:  # 存在y支配EPs[k]或者没有任何支配关系
                #     EPs.append(list(y))
                #     for delete_i in delete[::-1]:
                #         EPs.remove(delete_i)
        else:
            for i in range(len(ChrPops)):
                LS_choice = random.choices(LS_pool, weights=LS_weight)
                # print('pop_i',pop[i])
                y = LS_choice[0](ChrPops[i])
                t_y = target(y)
                newPop.append(list(y))
                ##更新z
                for j in range(len(Zmin[0])):
                    if t_y[j] < Zmin[0][j]:
                        Zmin[0][j] = t_y[j]
                ##更新外部存档
                ep = True  # 决定y是否放进EPs,True就放
                delete = []  # 装EPs个体
                for EP in EPs:  # EPs是所有支配个体，EPs外部存档
                    fun_EP = target(EP)
                    if isDominates(fun_EP, t_y) or fun_EP.all() == t_y.all():  # EPs[k]支配y
                        ep = False
                        break
                    elif isDominates(t_y, fun_EP):  # 存在y支配EPs[k]
                        delete.append(EP)  # 准备删除EPs[k]
                if ep:  # 存在y支配EPs[k]或者没有任何支配关系
                    EPs.append(list(y))
                    for delete_i in delete[::-1]:
                        EPs.remove(delete_i)
            targ_pop = target_pop(pop)
            newPop = np.array(newPop)
            targ_newpop = target_pop(newPop)
            pop, targ_pop = optSelect(pop,targ_pop,newPop,targ_newpop)

        if gen % 10 == 0:
            print("%d gen has completed!\n" % gen)


    Zmin = np.array(np.min(targ_pop, 0)).reshape(1, parse.nObj)  # 求理想点
    Zmax = np.array(np.max(targ_pop, 0)).reshape(1, parse.nObj)  # 求负理想点
    targ_pop = target_pop(pop)
    Target_Ep = target_pop(EPs)
    score_GD = GD(targ_pop, Target_Ep, Zmin, Zmax)
    score_IGD = IGD(targ_pop, Target_Ep, Zmin, Zmax)
    score_HV = HV(Target_Ep, Zmin, Zmax)
    return pop,targ_pop, score_GD, score_IGD, score_HV



if __name__ == '__main__':
    alg = IMOEAD  ###算法要改
    parse.terminate = 150
    pop, targ_pop, score_GD, score_IGD, score_HV = alg()
    index = np.argmin(targ_pop[:, 0])
    pop_i = pop[index]
    plot_figure_machine(pop_i)
    plot_figure_worker(pop_i)
    plot_fatig(pop_i)

    # for i in range(100):
    #     alg = IMOEAD  ###算法要改
    #     sheet = 0
    #     parse.terminate = 5*i
    #     col = 1 + i * (parse.nObj+1)
    #     pop, targ_pop, score_GD, score_IGD, score_HV = alg()
    #     Write_cell_str(sheet,1, col, str(alg))  ###列号要改
    #     Write_cell_str(sheet,2, col, "迭代次数: %s" % (parse.terminate))
    #     Write_cell_str(sheet,3, col, "score_GD: %s" % (score_GD))
    #     Write_cell_str(sheet,4, col, "score_IGD: %s" % (score_IGD))
    #     Write_cell_str(sheet,5, col, "score_HV: %s" % (score_HV))
    #     Write_cell_str(sheet,6, col, "targ_pop:")
    #     for i in range(len(targ_pop)):
    #         Write_cell(sheet, 7 + i , col, targ_pop[i])
    #
    #
    # for i in range(100):
    #     alg = MOEAD  ###算法要改
    #     sheet = 1
    #     parse.terminate = 5*i
    #     col = 1 + i * (parse.nObj+1)
    #     pop, targ_pop, score_GD, score_IGD, score_HV = alg()
    #
    #     Write_cell_str(sheet,1, col, str(alg))  ###列号要改
    #     Write_cell_str(sheet,2, col, "迭代次数: %s" % (parse.terminate))
    #     Write_cell_str(sheet,3, col, "score_GD: %s" % (score_GD))
    #     Write_cell_str(sheet,4, col, "score_IGD: %s" % (score_IGD))
    #     Write_cell_str(sheet,5, col, "score_HV: %s" % (score_HV))
    #     Write_cell_str(sheet,6, col, "targ_pop:")
    #     for i in range(len(targ_pop)):
    #         Write_cell(sheet, 7 + i , col, targ_pop[i])
    #
    #
    # for i in range(100):
    #     alg = NSGA2  ###算法要改
    #     sheet = 2
    #     parse.terminate = 5*i
    #     col = 1 + i * (parse.nObj+1)
    #     pop, targ_pop, score_GD, score_IGD, score_HV = alg()
    #
    #     Write_cell_str(sheet,1, col, str(alg))  ###列号要改
    #     Write_cell_str(sheet,2, col, "迭代次数: %s" % (parse.terminate))
    #     Write_cell_str(sheet,3, col, "score_GD: %s" % (score_GD))
    #     Write_cell_str(sheet,4, col, "score_IGD: %s" % (score_IGD))
    #     Write_cell_str(sheet,5, col, "score_HV: %s" % (score_HV))
    #     Write_cell_str(sheet,6, col, "targ_pop:")
    #     for i in range(len(targ_pop)):
    #         Write_cell(sheet, 7 + i , col, targ_pop[i])
    #     parse.terminate += 5
    #
    # for i in range(100):
    #     alg = SPEA2  ###算法要改
    #     sheet = 3
    #     parse.terminate = 5*i
    #     col = 1 + i * (parse.nObj+1)
    #     pop, targ_pop, score_GD, score_IGD, score_HV = alg()
    #
    #     Write_cell_str(sheet,1, col, str(alg))  ###列号要改
    #     Write_cell_str(sheet,2, col, "迭代次数: %s" % (parse.terminate))
    #     Write_cell_str(sheet,3, col, "score_GD: %s" % (score_GD))
    #     Write_cell_str(sheet,4, col, "score_IGD: %s" % (score_IGD))
    #     Write_cell_str(sheet,5, col, "score_HV: %s" % (score_HV))
    #     Write_cell_str(sheet,6, col, "targ_pop:")
    #     for i in range(len(targ_pop)):
    #         Write_cell(sheet, 7 + i , col, targ_pop[i])
    #     parse.terminate += 5




    # parse.terminate = 80
    # pop, targ_pop, score_GD, score_IGD, score_HV = alg()
    # index = np.argmin(targ_pop[:, 0])
    # pop_i = pop[index]
    # plot_figure_machine(pop_i)
    # plot_figure_worker(pop_i)
    # plot_fatig(pop_i)

    # ScatterPlot(targ_pop)
    # Cmax, Pc, Lb = np.mean(pop[:, 0]), np.mean(pop[:, 1]), np.mean(pop[:, 2])
    # Write_cell_str(line, col, str(alg))  ###列号要改
    # Write_cell_str(line + 1, col, "迭代次数: %s" % (parse.terminate))
    # Write_cell_str(line + 2, col, "score_GD: %s" % (score_GD))
    # Write_cell_str(line + 3, col, "score_IGD: %s" % (score_IGD))
    # Write_cell_str(line + 4, col, "score_HV: %s" % (score_HV))
    # Write_cell_str(line + 5, col, "targ_pop:")
    # for i in range(len(targ_pop)):
    #     Write_cell(line + i + 6, col, targ_pop[i])

    # for i in range(100):
    #     col = i + 1
    #     pop,targ_pop, score_GD, score_IGD, score_HV = alg()
    #     plot_figure(pop[0])
    #     plot_fatig(pop[0])
    #     ScatterPlot(targ_pop)
    #     Cmax,SI,Fmean = np.mean(pop[:,0]),np.mean(pop[:,1]),np.mean(pop[:,2])
    #     Write_cell(line, col, str(alg))  ###列号要改
    #     Write_cell(line + 1, col, "迭代次数: %s" % (parse.terminate))
    #     Write_cell(line + 2, col, "迭代次数: %s" % (parse.terminate))
    #     # Write_cell(line + 2, col, "score_GD: %s" % (score_GD))
    #     # Write_cell(line + 3, col, "score_IGD: %s" % (score_IGD))
    #     # Write_cell(line + 4, col, "score_HV: %s" % (score_HV))
    #     # for j in range(len(targ_pop)):
    #     #     Write_cell(line + 5 + j, col, targ_pop[j])
    #     print('pop:', targ_pop)
    #     print('targ_pop',targ_pop)
    #     print('algorithm',str(alg))
    #     print('迭代次数：',parse.terminate)
    #     print('score_GD:', score_GD)
    #     print('score_IGD:',score_IGD)
    #     print('score_HV',score_HV)
    #     parse.terminate += 10

    # 读取表格数据
    # open('result.xls', encoding='gbk')
    # data = pd.read_excel(io='./result.xls',sheet_name='NONE')
    # data = np.array(data)
    # data = data[6:65] #下面数据部分
    # # print(data)
    # operator = data[:, 0:3]
    # metropolis = data[:, 4]
    # tsea = data[:, 8]
    # moead = data[:, 12]
    # for i in range(len(tsea)):
    #     operator[i] = np.array(operator[i])
    #     metropolis[i] = np.array(metropolis[i])
    #     tsea[i] = np.array(tsea[i])
    #     moead[i] = np.array(moead[i])
    # a = imoead[0]
    # Compare_ScatterPlot(operator,metropolis,tsea,moead)
    # plot_box_diagram(moead,operator,metropolis,tsea)
    # plot_box_diagram(operator)







