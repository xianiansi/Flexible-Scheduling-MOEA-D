import copy
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib
import plotly as py
from colorama import init, Fore
from matplotlib import colors as mcolors
import plotly.figure_factory as ff
from matplotlib.pyplot import MultipleLocator
from collections import Counter
import openpyxl



# matplotlib.use('TKAgg')
pyplt = py.offline.plot
COLORS = list(mcolors.CSS4_COLORS)
LEN_COLORS = len(COLORS)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
init(autoreset=True)


class Nsga2():
    def __init__(self, pop_size, XOVR, MUTR, J, S, TM, JW, JmNumber, terminate, Es, u, fmax, lamda):  # var染色体,cost目标值
        self.pop_size = int(pop_size)
        self.nObj = 3  # 目标个数3
        self.XOVR = XOVR
        self.MUTR = MUTR
        self.J = J
        self.S = S
        self.W = J * S  # 3 * self.W  三层编码
        self.TM = TM
        self.JW = JW
        self.JmNumber = JmNumber
        self.terminate = int(terminate)
        self.Es = Es  # 工人个数
        self.u = u
        self.fmax = fmax
        self.lamda = lamda
        self.edgecolor, self.text_color = "black", "black"
        self.rotation, self.va, self.ha = "horizontal", "center", "center"
        self.alpha, self.height = 0.8, 0.8

    ##-----------------------------three-layer编码------------------------------------
    ############################# 三层编码初始化 #############################
    def initialize(self):
        Chrom = np.zeros((self.pop_size, 3 * self.W), dtype=int)  # 三层编码【工序】【机器】【人员】
        for j in range(self.pop_size):  # 第j条染色体
            # 工件序列OS
            OS = np.zeros(self.W, dtype=int)
            num = 0
            for job in range(self.J):
                for k in range(self.S):
                    OS[num] = job + 1  # OS=[ 1  1  1  2  2  2  3  3  3 ... 12 12 12]
                    num += 1
            np.random.shuffle(OS)  # 打乱
            Chrom[j, :self.W] = OS
            # 机器序列MS,人员序列WS
            MS = np.zeros(self.W, dtype=int)
            WS = np.zeros(self.W, dtype=int)
            for i in range(self.W):
                MS[i] = random.randint(1, self.JmNumber)  ##随机选择一个机器
                tmp = self.JW[MS[i] - 1]  ##机器对应的可选人员
                WS[i] = np.random.choice(tmp)  ##随机选择一个人员
            Chrom[j][self.W:2 * self.W] = MS
            Chrom[j][2 * self.W:3 * self.W] = WS
        return Chrom  # 初始化Chrom为一个种群

    ## 解码 第一层
    def decode(self, C):  # 染色体中的一个个体C
        q = len(C) // 3
        tmp = [0 for i in range(self.J)]  # 加工到第几个工序
        P = [0 for i in range(q)]
        for i in range(q):
            tmp[int(C[i]) - 1] += 1
            P[i] = C[i] * 100 + tmp[int(C[i]) - 1]
        return np.array(P)  # [101 301 ...] 解码后的染色体个体P （工件和工序）

    #############################目标值计算#####################################
    def caltimecost_fatigue(self, C):  # 编码染色体个体C
        fw = np.zeros((3000, self.Es))  # 记录每个人的疲劳度
        M = copy.deepcopy(C[self.J * self.S:self.J * self.S * 2])  # 机器编码
        E = copy.deepcopy(C[self.J * self.S * 2:self.J * self.S * 3])  # 人员编码
        PVal = np.zeros((2, self.W), dtype=int)  # 按染色体顺序记录每道工序加工所需时间、完成时刻
        TMF = np.zeros((4, self.JmNumber), dtype=int)  # 记录机器上次完工时刻、累计总加工时间、上次加工工序、上次加工时间(不含换模)
        TP = np.zeros((1, self.J), dtype=int)  # 记录每个工件的上次完工时刻
        TWF = np.zeros((2, self.Es), dtype=int)  # 记录人员上次完工时间、上次加工机器
        P = self.decode(C)
        for i in range(self.W):
            val = int(P[i])
            a = int(val % 100)  # 工序
            b = int((val - a) / 100)  # 工件
            mi = int(M[i])  # 机器序号
            ei = int(E[i])  # 加工该机器的人员
            t = self.TM[b - 1][a - 1][mi - 1]  # 对应的机器加工时间
            TMval = TMF[0][mi - 1]  # 机器上次的完工时刻
            TMpro = TMF[2][mi - 1]  # 机器上次加工的工件
            TMtime = TMF[3][mi - 1]  # 机器上次加工时间(不含换模)
            TPval = TP[0][b - 1]  # 工件上次完工时刻
            TWval = TWF[0][ei - 1]  # 工人上次完工时刻

            # 判断人员是否需要休息
            startval = max(TMval, TPval, TWval)
            while fw[int(startval) - 1][int(ei) - 1] >= self.fmax:  # 如果人员在开始前一秒还是疲劳度过高的话就休息gap
                gap = 5
                startval += gap

            ##计算空闲时间的疲劳度值变化
            TWval = TWF[0][ei - 1]  # 工人上一次完工时间
            for k in range(int(TWval), int(startval)):
                fw[k][int(ei) - 1] = fw[k - 1][int(ei) - 1] * (math.e ** (- self.u[int(ei) - 1]))

            if TMpro != a:  # 上次加工的工序和现在不同
                ChanTime = int(np.ceil(0.5 * (t + TMtime) * 0.5))  # 向上取整，本次换模时间
            else:
                ChanTime = 0

            TMF[3][mi - 1] = t  # 加工时间不含换模*用于计算下一次换模时间
            t += ChanTime  # 加工时间含换模
            PVal[0][i] = t  # 工件工序加工时长
            PVal[1][i] = startval + t  # 工件工序加工完成时刻
            TMF[0][mi - 1] = startval + t  # 机器加工完成时刻
            TMF[1][mi - 1] += t  # 机器加工总时长
            TMF[2][mi - 1] = a  # 机器加工工序
            TP[0][b - 1] = startval + t  # 工件加工完成时刻
            TWF[0][ei - 1] = startval + t  # 人员加工完成时刻
            TWF[1][ei - 1] = mi  # 人员加工机器

            ##计算加工工件时的疲劳度变化
            for j in range(int(startval), int(startval) + t):
                fw[j][int(ei) - 1] = 1 - (1 - fw[j - 1][int(ei) - 1]) * math.e ** (-self.lamda[int(ei) - 1])

            # 人员没有任务后疲劳度的变化情况
            if ei not in E[i + 1:]:
                for k in range(int(startval) + t, int(startval) + t + 100):
                    fw[k][int(ei) - 1] = fw[k - 1][int(ei) - 1] * (math.e ** (- self.u[int(ei) - 1]))

        # 计算三个目标值
        makespan = max(PVal[1][:])
        temp = np.zeros(self.Es)
        for i in range(self.Es):  # 每个员工加工过程中的疲劳度总和
            temp[i] = np.sum(fw[:makespan, i])
        FwTotal = np.sum(temp)
        MachineLoad = np.sum(TMF[1])  # 所有机器的总完成时间
        return PVal, fw, P, makespan, FwTotal, MachineLoad

    def target(self, spring):  # 计算某个体的3个目标函数
        targ = np.zeros(self.nObj)
        PVal, fw, P, makespan, FwTotal, MachineLoad = self.caltimecost_fatigue(spring)
        targ[0] = makespan
        targ[1] = FwTotal
        targ[2] = MachineLoad
        return targ

    def target_pop(self, Chrom):
        targ_pop = np.zeros((len(Chrom), self.nObj))
        for i in range(len(Chrom)):
            PVal, fw, P, makespan, FwTotal, MachineLoad = self.caltimecost_fatigue(Chrom[i])
            targ_pop[i, 0] = makespan
            targ_pop[i, 1] = FwTotal
            targ_pop[i, 2] = MachineLoad
        return targ_pop

    ##-----------------------------二元锦标赛选择------------------------------------
    def tournament_selection(self, Chrom, idx1, idx2):
        ranks = self.nonDominationSort(Chrom)
        distances = self.crowdingDistanceSort(Chrom, ranks)
        if ranks[idx1] < ranks[idx2]:
            idx = idx1
        elif ranks[idx1] > ranks[idx2]:
            idx = idx2
        else:
            if distances[idx1] <= distances[idx2]:
                idx = idx2
            else:
                idx = idx1
        return idx

    ############################非支配排序#################################
    def nonDominationSort(self, Chrom):
        targ = self.target_pop(Chrom)
        nPop = Chrom.shape[0]
        nF = targ.shape[1]  # 目标函数的个数
        ranks = np.zeros(nPop, dtype=np.int32)
        nPs = np.zeros(nPop)  # 每个个体p被支配解的个数
        sPs = []  # 每个个体支配的解的集合，把索引放进去
        for i in range(nPop):
            iSet = []  # 解i的支配解集
            for j in range(nPop):
                if i == j:
                    continue
                isDom1 = targ[i] <= targ[j]
                isDom2 = targ[i] < targ[j]
                # 是否支配该解-> i支配j
                if sum(isDom1) == nF and sum(isDom2) >= 1:
                    iSet.append(j)
                    # 是否被支配-> i被j支配
                if sum(~isDom2) == nF and sum(~isDom1) >= 1:
                    nPs[i] += 1
            sPs.append(iSet)  # 添加i支配的解的索引
        r = 0  # 当前等级为 0， 等级越低越好
        indices = np.arange(nPop)
        while sum(nPs == 0) != 0:
            rIdices = indices[nPs == 0]  # 当前被支配数为0的索引
            ranks[rIdices] = r
            for rIdx in rIdices:
                iSet = sPs[rIdx]
                nPs[iSet] -= 1
            nPs[rIdices] = -1  # 当前等级的被支配数设置为负数
            r += 1
        return ranks

    def isDominates(self, s1, s2):  # x是否支配y
        return (s1 <= s2).all() and (s1 < s2).any()
    ##-----------------------------染色体交叉------------------------------------
    ############################交叉、变异#################################
    def cross_mutation(self,Chrom):
        Chrom_new = copy.deepcopy(Chrom)
        # 交叉
        N = Chrom.shape[0]
        Chrom1 = Chrom[:int(N / 2)]
        Chrom2 = Chrom[int(N / 2):]
        for k in range(N):
            if random.random()<self.XOVR:
                s1 = Chrom1[np.random.choice(int(N / 2), 1)][0]
                s2 = Chrom2[np.random.choice(int(N / 2), 1)][0]
                temp = np.zeros(self.J, dtype=int)  # 子代中每个工件已经有几道工序了
                offspring = np.zeros(3 * self.W, dtype=int)
                at1 = 0  # parent1指针
                at2 = 0  # parent2指针
                at = True  # 从哪个parent复制
                for i in range(len(offspring) // 3):
                    while (offspring[i] == 0):  # 直到被赋值
                        if at:  # 从parent1取基因
                            if temp[s1[at1] - 1] < self.S:  # parent1对应的这个基因在子代中还没到达最大工序数
                                offspring[i] = s1[at1]  # 赋值
                                offspring[i + self.W] = s1[at1 + self.W]
                                offspring[i + self.W * 2] = s1[at2 + self.W * 2]
                            at1 += 1  # 不管是否赋值，at1指针向后一格
                            at = not at  # 逻辑取反，下次从parent2取
                        else:  # 从parent2取基因
                            if temp[s2[at2] - 1] < self.S:
                                offspring[i] = s2[at2]
                                offspring[i + self.W] = s2[at2 + self.W]
                                offspring[i + self.W * 2] = s2[at2 + self.W * 2]
                            at2 += 1
                            at = not at  # 逻辑取反
                    temp[offspring[i] - 1] += 1
                Chrom_new[k] = offspring
            else:
                Chrom_new[k] = Chrom[np.random.choice(N,1)]
        #变异
        for i in range(N):
            length = self.W  # len=36
            choice = np.random.choice(length,6,replace=False)
            if random.random()<self.MUTR:
                for j in range(len(choice)):
                    pos = random.randint(0, length - 1)
                    mi = random.randint(1,self.JmNumber)
                    Chrom_new[i][choice[j] + self.W] = mi
                    wList = self.JW[int(mi) - 1]  # 对应员工列表
                    Chrom_new[i][pos + self.W * 2] = np.random.choice(wList)  # 随机选择一个员工，可能出现员工没变的情况
            else:
                pass
        return Chrom_new


    def optSelect(self, pops, fits, chrPops, chrFits):
        nPop, nChr = pops.shape
        nF = fits.shape[1]
        newPops = np.zeros((nPop, nChr),dtype=int)
        newFits = np.zeros((nPop, nF))
        # 合并父代种群和子代种群构成一个新种群
        MergePops = np.concatenate((pops, chrPops), axis=0)  # 拼接
        MergeFits = np.concatenate((fits, chrFits), axis=0)
        MergeRanks = self.nonDominationSort(MergePops)  # 两个种群合并了，两倍大小的种群
        MergeDistances = self.crowdingDistanceSort(MergePops, MergeRanks)

        indices = np.arange(MergePops.shape[0])
        r = 0
        i = 0
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
        while i + len(rIndices) <= nPop:
            newPops[i:i + len(rIndices)] = MergePops[rIndices]  # 精英策略，先把等级小的全都放进新种群中
            newFits[i:i + len(rIndices)] = MergeFits[rIndices]
            r += 1  # 当前等级+1
            i += len(rIndices)
            rIndices = indices[MergeRanks == r]  # 当前等级为r的索引

        if i < nPop:  # 还有一部分没放，就在当前rank按拥挤度才弄个大到小放进去填满
            rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
            rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
            surIndices = rIndices[rSortedIdx[:(nPop - i)]]
            newPops[i:] = MergePops[surIndices]
            newFits[i:] = MergeFits[surIndices]
        return (newPops, newFits)

    ##-----------------------------拥挤度排序算法-----------------------------------
    def crowdingDistanceSort(self, Chrom, ranks):
        fits = np.zeros((Chrom.shape[0],3))
        for i in range(0, self.pop_size):
            PVal, fw, P, makespan, FwTotal, MachineLoad = self.caltimecost_fatigue(Chrom[i])
            fits[i][0] = makespan
            fits[i][1] = FwTotal
            fits[i][2] = MachineLoad
        nPop = Chrom.shape[0]
        nF = fits.shape[1]  # 目标个数
        dis = np.zeros(nPop)
        nR = ranks.max()  # 最大等级
        indices = np.arange(nPop)
        for r in range(nR + 1):
            rIdices = indices[ranks == r]  # 当前等级种群的索引
            rPops = Chrom[ranks == r]  # 当前等级的种群
            rFits = fits[ranks == r]  # 当前等级种群的适应度
            rSortIdices = np.argsort(rFits, axis=0)  # 对纵向排序的索引
            rSortFits = np.sort(rFits, axis=0)
            fMax = np.max(rFits, axis=0)
            fMin = np.min(rFits, axis=0)
            n = len(rIdices)
            for i in range(nF):
                orIdices = rIdices[rSortIdices[:, i]]  # 当前操作元素的原始位置
                j = 1
                while n > 2 and j < n - 1:
                    if fMax[i] != fMin[i]:
                        dis[orIdices[j]] += (rSortFits[j + 1, i] - rSortFits[j - 1, i]) / \
                                            (fMax[i] - fMin[i])
                    else:
                        dis[orIdices[j]] = np.inf
                    j += 1
                dis[orIdices[0]] = np.inf
                dis[orIdices[n - 1]] = np.inf
        return dis

    def IGD(self,popfun, PF):
        distance = np.min(self.EuclideanDistances(PF, popfun), 1)
        score = np.mean(distance)
        return score

    def EuclideanDistances(self,A, B):
        BT = B.transpose()
        # vecProd = A * BT
        vecProd = np.dot(A, BT)
        # print(vecProd)
        SqA = A ** 2
        # print(SqA)
        sumSqA = np.matrix(np.sum(SqA, axis=1))
        sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
        # print(sumSqAEx)

        SqB = B ** 2
        sumSqB = np.sum(SqB, axis=1)
        sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
        SqED = sumSqBEx + sumSqAEx - 2 * vecProd
        SqED[SqED < 0] = 0.0
        ED = np.sqrt(SqED)
        return ED

    def abtain_chrom(self):
        Pops = self.initialize()
        targ = self.target_pop(Pops)
        Zmin = np.array(np.min(targ, 0)).reshape(1, self.nObj)  # 求理想点（M个目标值 ）[1,1,1]三个目标的群体最小值
        iter = 0

        while iter < self.terminate:
            print("第{name}次迭代".format(name=iter))
            matingpool = random.sample(range(self.pop_size), self.pop_size)  # 就是把1-popsize-1打乱顺序
            ChrPops = self.cross_mutation(Pops[matingpool, :])
            chrTarg = self.target_pop(ChrPops)
            Pops, targ = self.optSelect(Pops, targ, ChrPops, chrTarg)

            mixpop = np.concatenate((Pops, ChrPops), axis=0)
            mixpopfun = self.target_pop(mixpop)
            Zmin = np.array(np.min(mixpopfun, 0)).reshape(1, self.nObj)  # 更新理想点
            iter += 1

        # IGD
        score = self.IGD(targ, Zmin)  # weights这里原本是PF，这里存疑，所谓真实的PF
        PVal, fw, P, makespan, FwTotal, MachineLoad = self.caltimecost_fatigue(np.array(Pops[0]))
        return PVal, fw, P, makespan, FwTotal, MachineLoad, Pops, score


    def Write_cell(self,line,column,string):
        wb = openpyxl.load_workbook("result.xlsx")  # 生成一个已存在的wookbook对象
        wb1 = wb.active  # 激活sheet
        wb1.cell(line, column, string)  # 对应14行B
        wb.save("result.xlsx")  # 保存

    def Write_excel(self, line, column):
        # ---------------数据保存-------------
        for i in range(20):
            start = time.time()
            PVal, fw, P, makespan, FwTotal, MachineLoad, Chrom, score = self.abtain_chrom()
            end = time.time()
            self.Write_cell(line + i, column, end - start)
            self.Write_cell(line + i, column + 1, makespan)
            self.Write_cell(line + i, column + 2, FwTotal)
            self.Write_cell(line + i, column + 3, MachineLoad)
            self.Write_cell(line + i, column + 4, score)

    ##-----------------------------画图------------------------------------
    def plot_figure(self):
        # ---------------甘特图-------------
        start = time.time()
        PVal, fw, P, makespan, FwTotal, MachineLoad, Pops, score = self.abtain_chrom()
        end = time.time()
        print("calculation time：%2f second" % (end - start))
        print("makespan", makespan)
        print('FwTotal', FwTotal)
        print('MachineLoad', MachineLoad)
        cst = PVal[1][:] - PVal[0][:]  # 每道工序开始加工时刻
        cpt = PVal[0][:]  # 每道工序的加工时长
        cft = PVal[1][:]  # 每道工序的加工完成时刻
        cmn = np.zeros(self.W, dtype=int)  # 对应的机器号
        cwn = np.zeros(self.W, dtype=int)  # 对应人员编号
        C = Pops[0]
        M = copy.deepcopy(C[self.J * self.S:self.J * self.S * 2])  # 机器编码
        E = copy.deepcopy(C[self.J * self.S * 2:self.J * self.S * 3])  # 人员编码
        for i in range(self.W):
            val = int(P[i])
            a = int(val % 100)  # 工序
            b = int((val - a) / 100)  # 工件
            mi = int(M[i])  # 机器序号
            ei = int(E[i])  # 人员序号
            cmn[i] = mi
            cwn[i] = ei
        osc = np.tile(C[0:self.W], 1)

        # ---------------甘特图-------------
        plt.figure()
        for i in range(self.W):
            if cft[i] != 0:
                plt.barh(y=cmn[i], width=cft[i] - cst[i], height=self.height, left=cst[i],
                         color=COLORS[osc[i] % LEN_COLORS], alpha=self.alpha, edgecolor=self.edgecolor)
                sf = r"$_{%s}$" % cst[i], r"$_{%s}$" % cft[i]
                x = cst[i], cft[i]
                for j, k in enumerate(sf):  # 时间刻度
                    plt.text(x=x[j], y=cmn[i] - self.height / 2, s=k,
                             rotation=self.rotation, va="top", ha=self.ha)
                text = r"${%s}$" % (osc[i])  # 编号
                text1 = r"${w%s}$" % (cwn[i])
                plt.text(x=cst[i] + 0.2 * cpt[i], y=cmn[i], s=text, c=self.text_color,
                         rotation=self.rotation, va=self.va, ha=self.ha)
                plt.text(x=cst[i] + 0.8 * cpt[i], y=cmn[i], s=text1, c=self.text_color,
                         rotation=self.rotation, va=self.va, ha=self.ha)
        plt.ylabel(r"Machine", fontsize=12, fontproperties="Arial")
        plt.xlabel(r"Makespan", fontsize=12, fontproperties="Arial")
        plt.title(r"Gantt Chart", fontsize=14, fontproperties="Arial")
        plt.show()

        #---------------甘特图-------------
        plt.figure()
        for i in range(self.W):
            if cft[i] != 0:
                plt.barh(y=cmn[i], width=cft[i] - cst[i], height=self.height, left=cst[i],
                         color=COLORS[osc[i] % LEN_COLORS], alpha=self.alpha, edgecolor=self.edgecolor)
                sf = r"$_{%s}$" % cst[i], r"$_{%s}$" % cft[i]
                x = cst[i], cft[i]
                for j, k in enumerate(sf):  # 时间刻度
                    plt.text(x=x[j], y=cmn[i] - self.height / 2, s=k,
                             rotation=self.rotation, va="top", ha=self.ha)
                text = r"${%s}$" % (osc[i])  # 编号
                text1 = r"${w%s}$" % (cwn[i])
                plt.text(x=cst[i] + 0.2 * cpt[i], y=cmn[i], s=text, c=self.text_color,
                         rotation=self.rotation, va=self.va, ha=self.ha)
                plt.text(x=cst[i] + 0.8 * cpt[i], y=cmn[i], s=text1, c=self.text_color,
                         rotation=self.rotation, va=self.va, ha=self.ha)

        plt.ylabel(r"Machine", fontsize=12, fontproperties="Arial")
        plt.xlabel(r"Makespan", fontsize=12, fontproperties="Arial")
        plt.title(r"Gantt Chart", fontsize=14, fontproperties="Arial")


        #-----------疲劳度曲线-----------
        plt.figure()
        fw = np.round(fw, 4)
        fw = fw[0:int(makespan) + 1, :]
        for z in range(len(fw[0])):
            fw[0][z] = 0
        x = range(0, int(makespan) + 1)
        y1 = fw[:, 0]
        y2 = fw[:, 1]
        y3 = fw[:, 2]
        y4 = fw[:, 3]
        y5 = fw[:, 4]
        y6 = [self.fmax for i in range(int(makespan) + 1)]
        plt.plot(x, y1, label='Worker 1', linewidth=2, color='lightskyblue', linestyle='solid')
        plt.plot(x, y2, label='Worker 2', linewidth=2, color='firebrick', linestyle='solid')
        plt.plot(x, y3, label='Worker 3', linewidth=2, color='thistle', linestyle='solid')
        plt.plot(x, y4, label='Worker 4', linewidth=2, color='purple', linestyle='solid')
        plt.plot(x, y5, label='Worker 5', linewidth=2, color='black', linestyle='solid')
        plt.plot(x, y6, label='ε', linewidth=2, color='gray')
        plt.xlabel('Time', fontsize=12, fontproperties="Arial")
        plt.ylabel('Fatigue', fontsize=12, fontproperties="Arial")
        plt.title('Fatigue Curve', fontsize=14, fontproperties="Arial")
        plt.legend()
        plt.show()

    def output_Chrom(self):
        PVal, fw, P, makespan, FwTotal, MachineLoad, Chrom, score = self.abtain_chrom()
        Popfun = self.target_pop(Chrom)
        return Popfun


if __name__ == "__main__":
    pop_size = 50
    terminate = 20
    XOVR = 0.5
    MUTR = 0.5
    S = 3
    JmNumber = 7
    # Es = 5
    fmax = 0.8
    # u = np.array([0.32, 0.26, 0.29, 0.32, 0.29])
    # lamda = np.array([0.147, 0.185, 0.216, 0.255, 0.258])
    # 每道工序在每个机器上的时间

    J = 12
    TM = np.array([[[7, 3, 4, 3, 9, 9, 4], [6, 6, 5, 4, 6, 5, 5], [15, 13, 14, 14, 13, 18, 13]],
                   [[3, 5, 7, 6, 5, 5, 9], [6, 6, 5, 4, 5, 5, 4], [11, 15, 17, 16, 15, 12, 10]],
                   [[9, 3, 3, 5, 5, 7, 9], [4, 5, 5, 5, 6, 5, 6], [13, 16, 13, 17, 8, 12, 15]],
                   [[3, 8, 9, 9, 6, 6, 8], [4, 5, 6, 4, 6, 6, 5], [12, 10, 15, 16, 14, 16, 8]],
                   [[9, 8, 3, 5, 5, 7, 7], [4, 5, 5, 5, 4, 4, 6], [15, 11, 14, 18, 14, 18, 18]],
                   [[6, 5, 5, 8, 5, 4, 6], [5, 4, 6, 6, 6, 4, 4], [8, 10, 13, 16, 12, 8, 12]],
                   [[7, 9, 3, 5, 6, 3, 4], [6, 6, 4, 5, 5, 6, 5], [11, 8, 13, 18, 16, 14, 17]],
                   [[7, 9, 3, 5, 6, 3, 4], [6, 5, 6, 6, 4, 6, 4], [10, 15, 17, 17, 12, 16, 15]],
                   [[6, 9, 9, 3, 5, 4, 9], [5, 6, 5, 6, 5, 5, 4], [15, 18, 10, 15, 16, 15, 11]],
                   [[7, 9, 6, 8, 7, 8, 3], [5, 4, 4, 4, 5, 6, 5], [16, 13, 10, 14, 11, 13, 9]],
                   [[9, 3, 9, 5, 7, 6, 4], [6, 6, 5, 5, 6, 6, 4], [9, 9, 18, 11, 18, 13, 12]],
                   [[4, 8, 3, 4, 5, 3, 9], [5, 4, 5, 4, 5, 6, 4], [10, 11, 9, 14, 15, 14, 13]]
                   ])
    # JW = np.array([[1, 2, 5], [1, 3, 4], [2, 3, 5], [1, 3, 4], [2, 4, 5], [1, 4, 5], [2, 3, 5]])
    Es = 3
    u = np.array([0.32, 0.26, 0.29])
    lamda = np.array([0.147, 0.185, 0.216])
    JW = np.array([[1, 2], [1, 3], [2, 3], [1, 3], [2, 3], [1, 2], [2, 3]])
    Schedule_strategy = Nsga2(pop_size, XOVR, MUTR, J, S, TM, JW, JmNumber, terminate, Es, u, fmax, lamda)
    # Schedule_strategy3.plot_figure()
    Schedule_strategy.Write_excel(76,11)

