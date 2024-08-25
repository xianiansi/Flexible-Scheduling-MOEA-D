import numpy as np
import os
import openpyxl
import math
import random
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from scipy import integrate


def update_archive(population, archive, archive_size):
    """
    更新外部存档

    参数：
    population -- 当前种群，每行表示一个解，每列表示一个目标函数值
    archive -- 外部存档，每行表示一个解，每列表示一个目标函数值
    archive_size -- 外部存档大小

    返回值：
    更新后的外部存档
    """
    # 合并当前种群和外部存档
    merged = np.concatenate((population, archive), axis=0)

    # 计算每个解的非劣等级
    ranks = calculate_nondominated_sort(merged)[0]

    # 每个等级保留的解数
    rank_sizes = [np.sum(ranks == i) for i in range(len(np.unique(ranks)))]

    # 从每个等级中选择一定数量的解加入到外部存档中
    new_archive = []
    for i, rank_size in enumerate(rank_sizes):
        if len(new_archive) + rank_size <= archive_size:
            # 如果当前等级的解数不超过外部存档的剩余容量，全部加入
            new_archive.extend(merged[ranks == i])
        else:
            # 如果当前等级的解数超过外部存档的剩余容量，按照拥挤度选择最优的一部分
            front = merged[ranks == i]
            distances = calculate_crowding_distance(front)
            sorted_indices = np.argsort(-distances)
            new_archive.extend(front[sorted_indices][:archive_size - len(new_archive)])

        # 如果外部存档已满，则不再处理后续等级
        if len(new_archive) >= archive_size:
            break

    # 返回更新后的外部存档
    return np.array(new_archive)

def moead(population, obj_func, weight_vectors, T, k, max_evaluations):
    """
    MOEA/D算法实现
    :param population: 种群
    :param obj_func: 目标函数
    :param weight_vectors: 权重向量
    :param T: 邻域大小
    :param k: 选择操作的参数
    :param max_evaluations: 最大评价次数
    :return: 外部解集
    """
    # 初始化邻居关系
    num_weight_vectors = len(weight_vectors)
    neighbor_table = np.zeros((num_weight_vectors, T), dtype=int)
    for i in range(num_weight_vectors):
        distances = np.sqrt(np.sum((weight_vectors - weight_vectors[i])**2, axis=1))
        neighbors = np.argsort(distances)[:T]
        neighbor_table[i] = neighbors

    # 初始化外部解集
    external_archive = []

    # 初始化评价次数
    num_evaluations = 0

    # 开始迭代
    while num_evaluations < max_evaluations:
        for i in range(num_weight_vectors):
            # 选择T个邻居
            neighbors = neighbor_table[i]

            # 从邻居中选择k个最好的个体
            k_best_indices = np.argsort([obj_func(population[j]) for j in neighbors])[:k]
            k_best_neighbors = neighbors[k_best_indices]

            # 选择一个父代个体
            parent = population[np.random.choice(k_best_neighbors)]

            # 对父代个体进行变异得到一个新个体
            child = parent + np.random.normal(size=parent.shape)

            # 计算新个体的适应度值
            child_fitness = obj_func(child)

            # 更新外部解集
            dominated = False
            dominated_indices = []
            for j, external_solution in enumerate(external_archive):
                if all(child_fitness <= external_solution): #新解不行，被支配了
                    dominated = True
                    break
                elif all(child_fitness >= external_solution): #EP某个解被支配了
                    dominated_indices.append(j)
            # 新解可以支配存档里某个解或者和它们是平行关系
            if not dominated:
                external_archive.append(child_fitness)
                for j in dominated_indices[::-1]:
                    del external_archive[j]

            # 更新种群
            for j in k_best_neighbors:  #新解可以替代这个weight下的所有邻居解，就成功替代
                if all(child_fitness <= obj_func(population[j])):
                    population[j] = child

            # 更新评价次数
            num_evaluations += 1

            # 判断是否达到最大评价次数
            if num_evaluations >= max_evaluations:
                break

    return external_archive

def Get_Problem(path):
    # 需要解决的问题
    # 传入的path为当前文档所在路径之后的拼接路径
    if os.path.exists(path):
        # 提取txt文件中的数据
        with open(path, 'r') as data:
            List = data.readlines()
            # print(List)
            P = []
            for line in List:
                list_ = []
                for j in range(len(line)):
                    try:
                        if line[j].isdigit() and line[j + 1].isdigit():
                            list_.append(int(line[j]) * 10 + int(line[j + 1]))
                            k = line[j]
                        if line[j - 1].isdigit() and line[j].isdigit() and j != 0:
                            pass

                        elif line[j].isdigit():
                            if line[j + 1].isspace() and j + 1 < len(line):
                                list_.append(int(line[j]))
                    except Exception as e:
                        print('可能出错，记得检查', e)
                        pass
                P.append(list_)
            for k in P:
                if k == []:
                    P.remove(k)
            # 将提取到的数据转化为具体的FJSP问题
            J_number = P[0][0]  # 工件数
            M_number = P[0][1]  # 机器数

            Problem = []
            for J_num in range(1, len(P)):
                O_num = P[J_num][0]  # 工件的工序数
                for Oij in range(O_num):
                    O_j = []
                    next = 1
                    while next < len(P[J_num]):
                        M_Oj = [0 for Oji in range(M_number)]
                        M_able = P[J_num][next]  # 加工第一道工序的可选机器数
                        able_set = P[J_num][next + 1:next + 1 + M_able * 2]
                        next = next + 1 + M_able * 2
                        for i_able in range(0, len(able_set), 2):
                            M_Oj[able_set[i_able] - 1] = able_set[i_able + 1]
                        O_j.append(M_Oj)
                Problem.append(O_j)
            for i_1 in range(len(Problem)):
                for j_1 in range(len(Problem[i_1])):
                    for k_1 in range(len(Problem[i_1][j_1])):
                        if Problem[i_1][j_1][k_1] == 0:
                            Problem[i_1][j_1][k_1] = 0
            # Max_len = []
            # for i_p in range(len(Problem)):
            #     Max_len.append(len(Problem[i_p]))
            # Max_Operation_len=max(Max_len)
            # for i_l in range(len(Problem)):
            #     if len(Problem[i_l])<Max_Operation_len:
            #         M_ky=[0 for Oji in range(M_number)]
            #         Problem_fake=Problem[i_l]
            #         Problem_fake.append(M_ky)
            #         Problem[i_l]=Problem_fake
            # Problem=np.array(Problem)
            # print(Problem)
    else:
        print('路径有问题')
    return Problem, J_number, M_number
def main1():
    Processing_time, J_num, M_num = Get_Problem('Mk01.fjs')
    print(Processing_time)
    print(J_num, M_num)


def topsis(X, w, is_benefit):
    # 首先定义了一个4行4列的矩阵X，代表4个候选方案的4个属性值。然后，我们定义了一个长度为4的权重向量w，代表每个属性的权重。
    # 接着，我们定义了一个长度为4的布尔向量is_benefit，用于表示每个属性是否为效益属性。如果为效益属性，则为True；否则为False。在本例中，我们假设所有属性均为效益属性。
    # 接下来，我们调用topsis函数来计算TOPSIS得分。该函数首先对矩阵进行标准化处理，然后计算加权标准化矩阵。接着，我们计算正理想解和负
    # 标准化矩阵
    X_norm = np.zeros(X.shape)
    for i in range(X.shape[1]):
        if is_benefit[i]:
            X_norm[:, i] = X[:, i] / np.max(X[:, i])
        else:
            X_norm[:, i] = np.min(X[:, i]) / X[:, i]
    # 计算加权标准化矩阵
    X_weighted = np.zeros(X.shape)
    for i in range(X.shape[1]):
        X_weighted[:, i] = X_norm[:, i] * w[i]
    # 计算正理想解和负理想解
    z_positive = np.zeros(X.shape[1])
    z_negative = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        if is_benefit[i]:
            z_positive[i] = np.max(X_weighted[:, i])
            z_negative[i] = np.min(X_weighted[:, i])
        else:
            z_positive[i] = np.min(X_weighted[:, i])
            z_negative[i] = np.max(X_weighted[:, i])
    # 计算距离正理想解和负理想解的欧几里得距离
    d_positive = np.sqrt(np.sum((X_weighted - z_positive)**2, axis=1))
    d_negative = np.sqrt(np.sum((X_weighted - z_negative)**2, axis=1))
    # 计算综合得分
    s = d_negative / (d_positive + d_negative)
    return s
def main2():
    X = np.array([[5, 7, 6, 4], [9, 8, 7, 6], [6, 7, 5, 3], [8, 6, 9, 5]])
    w = [0.25, 0.25, 0.25, 0.25]
    is_benefit = [True, True, True, True]
    # 计算TOPSIS得分
    scores = topsis(X, w, is_benefit)
    # 输出结果
    print("TOPSIS得分为：", scores)

def TOPSIS(targ_pop, lamda, is_benefit):
    # 首先定义了一个4行4列的矩阵X，代表4个候选方案的4个属性值。然后，我们定义了一个长度为4的权重向量w，代表每个属性的权重。
    # 接着，我们定义了一个长度为4的布尔向量is_benefit，用于表示每个属性是否为效益属性。如果为效益属性，则为True；否则为False。在本例中，我们假设所有属性均为效益属性。
    # 接下来，我们调用topsis函数来计算TOPSIS得分。该函数首先对矩阵进行标准化处理，然后计算加权标准化矩阵。接着，我们计算正理想解和负
    # 标准化矩阵
    X = targ_pop
    w = lamda
    X_norm = np.zeros(X.shape)
    for i in range(X.shape[1]):
        if is_benefit[i]:
            X_norm[:, i] = X[:, i] / np.max(X[:, i])
        else:
            X_norm[:, i] = np.min(X[:, i]) / X[:, i]
    # 计算加权标准化矩阵
    X_weighted = np.zeros(X.shape)
    for i in range(X.shape[1]):
        X_weighted[:, i] = X_norm[:, i] * w[i]
    # 计算正理想解和负理想解
    z_positive = np.zeros(X.shape[1])
    z_negative = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        if is_benefit[i]:
            z_positive[i] = np.max(X_weighted[:, i])
            z_negative[i] = np.min(X_weighted[:, i])
        else:
            z_positive[i] = np.min(X_weighted[:, i])
            z_negative[i] = np.max(X_weighted[:, i])
    # 计算距离正理想解和负理想解的欧几里得距离
    d_positive = np.sqrt(np.sum((X_weighted - z_positive)**2, axis=1))
    d_negative = np.sqrt(np.sum((X_weighted - z_negative)**2, axis=1))
    # 计算综合得分
    s = d_negative / (d_positive + d_negative)
    return s


def IGD(popfun,PF):
    num_PF = PF.shape[0]
    distances = np.zeros(num_PF)
    for i, p_true in enumerate(PF):
        print(p_true)
        print(popfun - p_true)
        distances[i] = np.min(np.linalg.norm(popfun - p_true, axis=1))
        print(distances[i])
    igd = np.sum(distances) / num_PF
    return igd
popfun = np.array([[1,1,1],[2,2,2]])
PF = np.array([[0,0,0],[0,1,0]])

def f(x, a, b):
    return a * x + b




class SPEA2():
    def __init__(self, dim, pop, max_iter):  # 维度，群体数量，迭代次数
        self.pc = 0.4  # 交叉概率
        self.pm = 0.4  # 变异概率
        self.dim = dim  # 搜索维度
        self.pop = pop  # 粒子数量
        self.K = int(np.sqrt(pop + pop))  # 距离排序，第k个距离值
        self.max_iter = max_iter  # 迭代次数
        self.population = []  # 父代种群
        self.archive = []  # 存档集合
        self.popu_arch = []  # 合并后的父代与存档集合种群
        # self.fronts = []                        #Pareto前沿面
        self.KNN = []  # 最近领域距离，K-th
        # self.rank = []#np.zeros(self.pop)       #非支配排序等级
        self.S = []  # 个体 i的 Strength Value
        self.D = []  # density，距离度量
        self.R = []  # 支配关系度量
        self.F = []  # 适应度
        self.objectives = []  # 目标函数值
        # self.np = []                            #该个体支配的其它个体数目
        self.set = []  # 被支配的个体集

    def init_Population(self):  # 初始化种群
        self.population = np.zeros((self.pop, self.dim))
        self.archive = np.zeros((self.pop, self.dim))
        for i in range(self.pop):
            for j in range(self.dim):
                self.population[i][j] = random.random()
                self.archive[i][j] = random.random()

    def popu_archive(self):  # Population和 Archive合并,pop*2
        self.popu_arch = np.zeros((2 * self.pop, self.dim))
        for i in range(self.pop):
            for j in range(self.dim):
                self.popu_arch[i][j] = self.population[i][j]
                self.popu_arch[i + self.pop][j] = self.archive[i][j]

    def cal_obj(self, position):  # 计算一个个体的多目标函数值 f1,f2 最小值
        f1 = position[0]
        f = 0
        for i in range(self.dim - 1):
            f += 9 * (position[i + 1] / (self.dim - 1))
        g = 1 + f
        f2 = g * (1 - np.square(f1 / g))
        return [f1, f2]

    def cal_fitness(self):  # 计算 Pt和 Et 适应度, F(i) = R(i) + D(i)
        self.objectives = []
        self.set = []
        self.S = np.zeros(2 * self.pop)
        self.D = np.zeros(2 * self.pop)
        self.R = np.zeros(2 * self.pop)
        self.F = np.zeros(2 * self.pop)
        self.KNN = np.zeros(2 * self.pop)
        position = []
        for i in range(2 * self.pop):
            position = self.popu_arch[i] #获得染色体
            self.objectives.append(self.cal_obj(position))  #每个染色体的目标值
        # 计算 S 值
        for i in range(2 * self.pop):
            temp = []
            for j in range(2 * self.pop):
                if j != i:
                    if self.objectives[i][0] <= self.objectives[j][0] and self.objectives[i][1] <= self.objectives[j][1]:
                        self.S[i] += 1  # i支配 j，np+1
                    if self.objectives[j][0] <= self.objectives[i][0] and self.objectives[j][1] <= self.objectives[i][1]:
                        temp.append(j)  # j支配 i
            self.set.append(temp)
            # 计算 R 值
        for i in range(2 * self.pop):
            for j in range(len(self.set[i])):
                self.R[i] += self.S[self.set[i][j]]
        # 计算 D 值
        for i in range(2 * self.pop):
            distance = []
            for j in range(2 * self.pop):
                if j != i:
                    distance.append(np.sqrt(np.square(self.objectives[i][0] - self.objectives[j][0]) + np.square(
                        self.objectives[i][1] - self.objectives[j][1])))
            distance = sorted(distance)
            self.KNN[i] = distance[self.K - 1]  # 其它个体与个体 i 的距离，升序排序，取第 K 个距离值
            self.D[i] = 1 / (self.KNN[i] + 2)
        # 计算 F 值
        for i in range(2 * self.pop):
            self.F[i] = self.D[i] + self.R[i]

    def update(self):  # 下一代 Archive
        # self.archive = []
        juli = []
        shiyingzhi = []
        a = 0
        for i in range(2 * self.pop):
            if self.F[i] < 1:
                juli.append([self.D[i], i])
                a = a + 1
            else:
                shiyingzhi.append([self.F[i], i])
        # 判断是否超出范围
        if a > self.pop:  # 截断策略
            juli = sorted(juli)
            for i in range(self.pop):
                self.archive[i] = self.popu_arch[juli[i][1]]
        if a == self.pop:  # 刚好填充
            for i in range(self.pop):
                self.archive[i] = self.popu_arch[juli[i][1]]
        if a < self.pop:  # 适应值筛选
            shiyingzhi = sorted(shiyingzhi)
            for i in range(a):
                self.archive[i] = self.popu_arch[juli[i][1]]
            for i in range(self.pop - a):
                self.archive[i + a] = self.popu_arch[shiyingzhi[i][1]]

    def cal_fitness2(self):  # 计算 Pt和 Et 适应度, F(i) = R(i) + D(i)
        self.objectives = []
        self.set = []
        self.S = np.zeros(self.pop)
        self.D = np.zeros(self.pop)
        self.R = np.zeros(self.pop)
        self.F = np.zeros(self.pop)
        self.KNN = np.zeros(self.pop)
        position = []
        for i in range(self.pop):
            position = self.archive[i]
            self.objectives.append(self.cal_obj(position))
        # 计算 S 值
        for i in range(self.pop):
            temp = []
            for j in range(self.pop):
                if j != i:
                    if self.objectives[i][0] <= self.objectives[j][0] and self.objectives[i][1] <= self.objectives[j][
                        1]:
                        self.S[i] += 1  # i支配 j，np+1
                    if self.objectives[j][0] <= self.objectives[i][0] and self.objectives[j][1] <= self.objectives[i][
                        1]:
                        temp.append(j)  # j支配 i
            self.set.append(temp)
            # 计算 R 值
        for i in range(self.pop):
            for j in range(len(self.set[i])):
                self.R[i] += self.S[self.set[i][j]]
        # 计算 D 值
        for i in range(self.pop):
            distance = []
            for j in range(self.pop):
                if j != i:
                    distance.append(np.sqrt(np.square(self.objectives[i][0] - self.objectives[j][0]) + np.square(
                        self.objectives[i][1] - self.objectives[j][1])))
            distance = sorted(distance)
            self.KNN[i] = distance[self.K - 1]  # 其它个体与个体 i 的距离，升序排序，取第 K 个距离值
            self.D[i] = 1 / (self.KNN[i] + 2)
        # 计算 F 值
        for i in range(self.pop):
            self.F[i] = self.D[i] + self.R[i]  # 适应度越小越好

    def selection(self):  # 轮盘赌选择
        pi = np.zeros(self.pop)  # 个体的概率
        qi = np.zeros(self.pop + 1)  # 个体的累积概率
        P = 0
        self.cal_fitness2()  # 计算Archive的适应值
        for i in range(self.pop):
            P += 1 / self.F[i]  # 取倒数，求累积适应度
        for i in range(self.pop):
            pi[i] = (1 / self.F[i]) / P  # 个体遗传到下一代的概率
        for i in range(self.pop):
            qi[0] = 0
            qi[i + 1] = np.sum(pi[0:i + 1])  # 累积概率
        # 生成新的 population
        self.population = np.zeros((self.pop, self.dim))
        for i in range(self.pop):
            r = random.random()  # 生成随机数，
            for j in range(self.pop):
                if r > qi[j] and r < qi[j + 1]:
                    self.population[i] = self.archive[j]
                j += 1

    def crossover(self):  # 交叉,SBX交叉
        for i in range(self.pop - 1):
            # temp1 = []
            # temp2 = []
            if random.random() < self.pc:
                # pc_point = random.randint(0,self.dim-1)        #生成交叉点
                # temp1.append(self.population[i][pc_point:self.dim])
                # temp2.append(self.population[i+1][pc_point:self.dim])
                # self.population[i][pc_point:self.dim] = temp2
                # self.population[i+1][pc_point:self.dim] = temp1
                a = random.random()
                for j in range(self.dim):
                    self.population[i][j] = a * self.population[i][j] + (1 - a) * self.population[i + 1][j]
                    self.population[i + 1][j] = a * self.population[i + 1][j] + (1 - a) * self.population[i][j]
            i += 2

    def mutation(self):  # 变异
        for i in range(self.pop):
            for j in range(self.dim):
                if random.random() < self.pm:
                    self.population[i][j] = self.population[i][j] - 0.1 + np.random.random() * 0.2
                    if self.population[i][j] < 0:
                        self.population[i][j] = 0  # 最小值0
                    if self.population[i][j] > 1:
                        self.population[i][j] = 1  # 最大值1

    def draw(self):  # 画图
        self.cal_fitness2()
        self.objectives = []
        a = 0
        for i in range(self.pop):
            if self.F[i] < 1:  # 非支配个体
                a += 1
                position = self.archive[i]
                self.objectives.append(self.cal_obj(position))
        x = []
        y = []
        for i in range(a):
            x.append(self.objectives[i][0])
            y.append(self.objectives[i][1])
        ax = plt.subplot(111)
        plt.scatter(x, y)  # ,marker='+')#self.objectives[:][0],self.objectives[:][1]) #?
        # plt.plot(,'--',label='')
        plt.axis([0.0, 1.0, 0.0, 1.1])
        xmajorLocator = MultipleLocator(0.1)
        ymajorLocator = MultipleLocator(0.1)
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.yaxis.set_major_locator(ymajorLocator)
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.title('ZDT2 Pareto Front')
        plt.grid()
        # plt.show()
        plt.savefig('SPEA ZDT2 Pareto Front 2.png')

    def run(self):  # 主程序
        self.init_Population()  # 初始化种群，选择交叉变异，生成子代种群
        self.popu_archive()
        self.cal_fitness()
        self.update()
        for i in range(self.max_iter - 1):
            self.selection()
            self.crossover()
            self.mutation()
            self.popu_archive()
            self.cal_fitness()
            self.update()
        self.draw()

def func(x):
    return 2*x  # 值得注意的是，一定要将表达式return出来

if __name__ == '__main__':
    SPEA = SPEA2(30, 100, 500)
    SPEA.run()


