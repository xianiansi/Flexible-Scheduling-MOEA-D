import copy
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import plotly as py
from colorama import init, Fore
from matplotlib import colors as mcolors
import time
import plotly.figure_factory as ff
from matplotlib.pyplot import MultipleLocator
from collections import Counter
import matplotlib as mpl
import openpyxl
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from scipy.special import comb  # comb 组合数Cmn
from itertools import combinations
from threading import Thread
from time import sleep, ctime
from utils import NSGA_vector, Estimate, AggFun, Select, Initial, Evolution, Search
import utils


matplotlib.use('TKAgg')
pyplt = py.offline.plot
color = [
    'lightcoral',
    'coral',
    'darkorange',
    'gold',
    'palegreen',
    'paleturquoise',
    'skyblue',
    'plum',
    'hotpink',
    'pink',
    'tomato',
    'lightgreen',
    'lightskyblue']
newcmp = mpl.colors.LinearSegmentedColormap.from_list(name='chaos', colors=color, N=12)  # N 表示插值后的颜色数目
color_list = [[newcmp(i)[0], newcmp(i)[1], newcmp(i)[2]] for i in range(newcmp.N)]
COLORS = list(color_list)
LEN_COLORS = len(COLORS)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
init(autoreset=True)
Esti, AggF, Sel, Init, Evol, Sea = Estimate(), AggFun(), Select(), Initial(), Evolution(), Search()  # 创建对象


class MOEA_D1():
    def __init__(self, pop_size, XOVR, MUTR, J, S, TM, JW, JmNumber, terminate, Es, u, fmax, lamda,
                 archive):  # var染色体,cost目标值
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
        self.archive = archive  # 存档数目
        self.edgecolor, self.text_color = "black", "black"
        self.rotation, self.va, self.ha = "horizontal", "center", "center"
        self.alpha, self.height = 0.8, 0.8

    ## 解码 第一层
    def decode(self, C):  # 染色体中的一个个体C
        q = len(C) // 3
        tmp = [0 for i in range(self.J)]  # 加工到第几个工序
        P = [0 for i in range(q)]
        for i in range(q):
            tmp[int(C[i]) - 1] += 1
            P[i] = C[i] * 100 + tmp[int(C[i]) - 1]
        return np.array(P)  # [101 301 ...] 解码后的染色体个体P （工件和工序）

    def decode_OS(self, OS):
        tmp = [0 for i in range(self.J)]  # 加工到第几个工序
        P = [0 for i in range(len(OS))]
        for i in range(len(OS)):
            tmp[int(OS[i]) - 1] += 1
            P[i] = OS[i] * 100 + tmp[int(OS[i]) - 1]
        a = np.zeros(self.W, dtype=int)
        b = np.zeros(self.W, dtype=int)
        for i in range(self.W):
            val = int(P[i])
            a[i] = int(val % 100)  # 工序
            b[i] = int((val - a[i]) / 100)  # 工件
        return a, b

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

    # 加了一个返回值（为变邻域做准本）
    def caltimecost_fatigue1(self, C):  # 编码染色体个体C
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
        return PVal, fw, P, makespan, FwTotal, MachineLoad, TMF[1], temp

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

    def Write_cell(self, line, column, string):
        wb = openpyxl.load_workbook("result.xlsx")  # 生成一个已存在的wookbook对象
        wb1 = wb.active  # 激活sheet
        wb1.cell(line, column, string)  # 对应14行B
        wb.save("result.xlsx")  # 保存

    def Write_excel(self, line, column):
        # ---------------数据保存-------------
        for i in range(20):
            start = time.time()
            PVal, fw, P, makespan, FwTotal, MachineLoad, Chrom, z, popfun, score = self.abtain_chrom()
            end = time.time()
            self.Write_cell(line + i, column, end - start)
            self.Write_cell(line + i, column + 1, makespan)
            self.Write_cell(line + i, column + 2, FwTotal)
            self.Write_cell(line + i, column + 3, MachineLoad)
            self.Write_cell(line + i, column + 4, score)

    def output_Chrom(self):
        PVal, fw, P, makespan, FwTotal, MachineLoad, Chrom, z, popfun, score = self.abtain_chrom()
        Popfun = self.target_pop(Chrom)
        return Popfun

    def plot_figure(self):
        # ---------------甘特图-------------
        start = time.time()
        PVal, fw, P, makespan, FwTotal, MachineLoad, Chrom, z, popfun, score = self.abtain_chrom()
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
        C = Chrom[0]
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
            # if random.random() < 0.5:
            #     MS = Init.prioc_MS(OS, self.decode_OS, self.W, self.JmNumber)
            # else:
            MS = Init.rand_MS(self.W, self.JmNumber)
            # if random.random() < 0.5:
            #     WS = Init.prio_WS(MS, self.W, self.JW, self.u)
            # else:
            WS = Init.rand_WS(MS, self.W, self.JW)
            Chrom[j][self.W:2 * self.W] = MS
            Chrom[j][2 * self.W:3 * self.W] = WS
        return Chrom  # 初始化Chrom为一个种群

    def plot_fatig(self):
        ##########疲劳度曲线#########
        plt.figure()
        PVal, fw, P, makespan, FwTotal, MachineLoad, Chrom, z, popfun, score = self.abtain_chrom()
        fw = np.round(fw, 4)
        fw = fw[:int(makespan) + 1, :]
        x = range(0, int(makespan) + 1)
        y1 = fw[:, 0]
        y2 = fw[:, 1]
        y3 = fw[:, 2]
        y4 = fw[:, 3]
        y5 = fw[:, 4]
        plt.plot(x, y1, label='Worker 1', linewidth=3, color='darkorange')
        plt.plot(x, y2, label='Worker 2', linewidth=3, color='pink')
        plt.plot(x, y3, label='Worker 3', linewidth=3, color='tomato')
        plt.plot(x, y4, label='Worker 4', linewidth=3, color='lightgreen')
        plt.plot(x, y5, label='Worker 5', linewidth=3, color='lightskyblue')
        plt.xlabel('Time', fontsize=14, fontproperties="Arial")
        plt.ylabel('Fatigue', fontsize=14, fontproperties="Arial")
        plt.title('Fatigue Curve', fontsize=14, fontproperties="Arial")
        plt.legend()
        plt.show()

    def abtain_chrom(self):
        T = [5, 10, 15, 20, 25, 30]  # 参数池
        T_score = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        LS = [Sea.LS1, Sea.LS2, Sea.LS3, Sea.LS4, Sea.LS5, Sea.LS6]  # 变邻域动作池
        gamma = 0.8
        Q_LS = np.zeros((self.nObj, len(LS)), dtype=int)

        Pop = self.initialize()
        PopFun = self.target_pop(Pop)  # Points
        # self.CompareFig(PopFun)
        ranks = Sel.nonDominationSort(PopFun)
        Points = PopFun[ranks == 0]
        EPs = copy.deepcopy([list(Pop[i]) for i in range(len(Pop))])
        rk = 0
        while len(Points) < 10:
            rk += 1
            Points = np.append(PopFun, PopFun[ranks == rk], axis=0)
        z = np.min(PopFun, axis=0)
        dc = 30
        vi = utils.Kmeans_vector(len(Pop), dc)  # R参数等于种群数量
        Lambda, sp_neighbors = vi.genvector1(Points, self.target, Pop, len(Pop))
        iter = 0
        print(len(EPs))
        while iter < self.terminate:
            print("第{name}次迭代".format(name=iter))
            epsi = iter / self.terminate  # 一开始是探索为主，后面是做测试为主
            # 对每一个训练,随机选择一种T
            if random.random() < epsi:
                T_choice = random.randint(0, 5)
            else:
                T_choice = Sel.roulette(T_score)
            T_i = T[T_choice]
            self.XOVR = 0.2 + epsi * 0.7
            self.MUTR = 0.2 + epsi * 0.7
            # T_i = min(15, len(Pop))
            sp_neighbors_ = sp_neighbors[:, :T_i]
            EP_last = EPs
            for i in range(len(Pop)):
                Bi = sp_neighbors_[i]  # 第i个个体的15个邻居（种群序号）
                choice = np.random.choice(T_i, 2, replace=False)  # 从0-14，选2个不重复邻居在Bi中的顺序号码
                k = Bi[choice[0]]  # 邻居序号1
                l = Bi[choice[1]]  # 邻居序号2
                xk = Pop[k]
                xl = Pop[l]
                y = Evol.cross_mutation1(xk, xl, self.MUTR, self.XOVR, self.J, self.W, self.S, self.JmNumber, self.JW)
                fv_y = np.array(self.target(y))  # 目标值
                # 更新z
                t = z > fv_y  # fv_y更优的目标为True
                z[t] = fv_y[t]  # z存储fv_y更优的目标
                # 更新邻域解
                # AggF.Tcheb_sum1(self.target, self.target_pop, Pop, y, Bi, z, Lambda)
                # AggF.isdominates_all2(y,EPs,self.target,self.archive)
                AggF.Tcheb_sum(self.target, Pop, y, Bi, z, Lambda)  # 个体i的邻居是否被y代替，更新种群(这个邻居就是种群的一个个体)
                AggF.isdominates_all(y, EPs, self.target, self.archive)
            iter += 1
            EP_now = EPs
            EP_last_fun = self.target_pop(EP_last)
            EP_now_fun = self.target_pop(EP_now)
            if Esti.Coverage(EP_last_fun, EP_now_fun) < Esti.Coverage(EP_now_fun, EP_last_fun):
                T_score[T_choice] += 0.5
            else:
                pass

        # fun_EPs = list(self.target_pop(np.array(EPs)))
        fun_EPs = copy.deepcopy([list(self.target(ep)) for ep in EPs])
        # print('fun_EPs',fun_EPs)
        # 变邻域搜索
        for s in EPs:
            PVal, fw, P, makespan, FwTotal, MachineLoad, TMF, fatigue = self.caltimecost_fatigue1(np.array(s))
            for j in range(100):
                s_new1, s_new2, s_new3, s_new4, s_new5, s_new6 = \
                    Sea.LS1(s, self.W), Sea.LS2(s, self.W), Sea.LS3(s, self.W, self.JmNumber), \
                    Sea.LS4(s, self.W, self.JW), Sea.LS5(s, self.J, self.W, TMF), Sea.LS6(s, self.J, self.W, fatigue)
                AggF.isdominates_all1(s_new1, EPs, self.target, fun_EPs, self.archive)
                AggF.isdominates_all1(s_new2, EPs, self.target, fun_EPs, self.archive)
                AggF.isdominates_all1(s_new3, EPs, self.target, fun_EPs, self.archive)
                AggF.isdominates_all1(s_new4, EPs, self.target, fun_EPs, self.archive)
                AggF.isdominates_all1(s_new5, EPs, self.target, fun_EPs, self.archive)
                AggF.isdominates_all1(s_new6, EPs, self.target, fun_EPs, self.archive)
        random.shuffle(EPs)
        # PVal, fw, P, makespan, FwTotal, MachineLoad = self.caltimecost_fatigue(np.array(EPs[0]))
        PVal, fw, P, makespan, FwTotal, MachineLoad, TMF, fatigue = self.caltimecost_fatigue1(np.array(EPs[0]))
        popfun = self.target_pop(EPs)
        z = np.array(z).reshape(1, self.nObj)
        score = Esti.IGD(popfun, Lambda)
        return PVal, fw, P, makespan, FwTotal, MachineLoad, EPs, z, popfun, score

    def ScatterPlot(self, A):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], marker="o", color="y")

        ax.set_xlabel('MakeSpan')
        ax.set_ylabel('SI')
        ax.set_zlabel('Fmean')  # 给三个坐标轴注明坐标名称
        plt.title("Three objectives ScatterPlot")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    pop_size = 50
    terminate = 20
    XOVR = 0.2
    MUTR = 0.2
    S = 3
    JmNumber = 7
    Es = 5
    fmax = 0.8
    u = np.array([0.32, 0.26, 0.29, 0.32, 0.29])
    # lamda = np.array([0.147, 0.185, 0.216, 0.255, 0.258])

    # u = np.array([0.33, 0.13, 0.60, 0.25, 0.46])
    lamda = np.array([0.147, 0.185, 0.216, 0.255, 0.258])

    archive = 50  # 存档数目
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
                   [[4, 8, 3, 4, 5, 3, 9], [5, 4, 5, 4, 5, 6, 4], [10, 11, 9, 14, 15, 14, 13]]])

    # J = 36
    # TM = np.array([[[7, 3, 4, 3, 9, 9, 4], [6, 6, 5, 4, 6, 5, 5], [15, 13, 14, 14, 13, 18, 13]],
    #                [[3, 5, 7, 6, 5, 5, 9], [6, 6, 5, 4, 5, 5, 4], [11, 15, 17, 16, 15, 12, 10]],
    #                [[9, 3, 3, 5, 5, 7, 9], [4, 5, 5, 5, 6, 5, 6], [13, 16, 13, 17, 8, 12, 15]],
    #                [[3, 8, 9, 9, 6, 6, 8], [4, 5, 6, 4, 6, 6, 5], [12, 10, 15, 16, 14, 16, 8]],
    #                [[9, 8, 3, 5, 5, 7, 7], [4, 5, 5, 5, 4, 4, 6], [15, 11, 14, 18, 14, 18, 18]],
    #                [[6, 5, 5, 8, 5, 4, 6], [5, 4, 6, 6, 6, 4, 4], [8, 10, 13, 16, 12, 8, 12]],
    #                [[7, 9, 3, 5, 6, 3, 4], [6, 6, 4, 5, 5, 6, 5], [11, 8, 13, 18, 16, 14, 17]],
    #                [[7, 9, 3, 5, 6, 3, 4], [6, 5, 6, 6, 4, 6, 4], [10, 15, 17, 17, 12, 16, 15]],
    #                [[6, 9, 9, 3, 5, 4, 9], [5, 6, 5, 6, 5, 5, 4], [15, 18, 10, 15, 16, 15, 11]],
    #                [[7, 9, 6, 8, 7, 8, 3], [5, 4, 4, 4, 5, 6, 5], [16, 13, 10, 14, 11, 13, 9]],
    #                [[9, 3, 9, 5, 7, 6, 4], [6, 6, 5, 5, 6, 6, 4], [9, 9, 18, 11, 18, 13, 12]],
    #                [[4, 8, 3, 4, 5, 3, 9], [5, 4, 5, 4, 5, 6, 4], [10, 11, 9, 14, 15, 14, 13]],
    #
    #                [[7, 3, 4, 3, 9, 9, 4], [6, 6, 5, 4, 6, 5, 5], [15, 13, 14, 14, 13, 18, 13]],
    #                [[3, 5, 7, 6, 5, 5, 9], [6, 6, 5, 4, 5, 5, 4], [11, 15, 17, 16, 15, 12, 10]],
    #                [[9, 3, 3, 5, 5, 7, 9], [4, 5, 5, 5, 6, 5, 6], [13, 16, 13, 17, 8, 12, 15]],
    #                [[3, 8, 9, 9, 6, 6, 8], [4, 5, 6, 4, 6, 6, 5], [12, 10, 15, 16, 14, 16, 8]],
    #                [[9, 8, 3, 5, 5, 7, 7], [4, 5, 5, 5, 4, 4, 6], [15, 11, 14, 18, 14, 18, 18]],
    #                [[6, 5, 5, 8, 5, 4, 6], [5, 4, 6, 6, 6, 4, 4], [8, 10, 13, 16, 12, 8, 12]],
    #                [[7, 9, 3, 5, 6, 3, 4], [6, 6, 4, 5, 5, 6, 5], [11, 8, 13, 18, 16, 14, 17]],
    #                [[7, 9, 3, 5, 6, 3, 4], [6, 5, 6, 6, 4, 6, 4], [10, 15, 17, 17, 12, 16, 15]],
    #                [[6, 9, 9, 3, 5, 4, 9], [5, 6, 5, 6, 5, 5, 4], [15, 18, 10, 15, 16, 15, 11]],
    #                [[7, 9, 6, 8, 7, 8, 3], [5, 4, 4, 4, 5, 6, 5], [16, 13, 10, 14, 11, 13, 9]],
    #                [[9, 3, 9, 5, 7, 6, 4], [6, 6, 5, 5, 6, 6, 4], [9, 9, 18, 11, 18, 13, 12]],
    #                [[4, 8, 3, 4, 5, 3, 9], [5, 4, 5, 4, 5, 6, 4], [10, 11, 9, 14, 15, 14, 13]],
    #
    #                [[7, 3, 4, 3, 9, 9, 4], [6, 6, 5, 4, 6, 5, 5], [15, 13, 14, 14, 13, 18, 13]],
    #                [[3, 5, 7, 6, 5, 5, 9], [6, 6, 5, 4, 5, 5, 4], [11, 15, 17, 16, 15, 12, 10]],
    #                [[9, 3, 3, 5, 5, 7, 9], [4, 5, 5, 5, 6, 5, 6], [13, 16, 13, 17, 8, 12, 15]],
    #                [[3, 8, 9, 9, 6, 6, 8], [4, 5, 6, 4, 6, 6, 5], [12, 10, 15, 16, 14, 16, 8]],
    #                [[9, 8, 3, 5, 5, 7, 7], [4, 5, 5, 5, 4, 4, 6], [15, 11, 14, 18, 14, 18, 18]],
    #                [[6, 5, 5, 8, 5, 4, 6], [5, 4, 6, 6, 6, 4, 4], [8, 10, 13, 16, 12, 8, 12]],
    #                [[7, 9, 3, 5, 6, 3, 4], [6, 6, 4, 5, 5, 6, 5], [11, 8, 13, 18, 16, 14, 17]],
    #                [[7, 9, 3, 5, 6, 3, 4], [6, 5, 6, 6, 4, 6, 4], [10, 15, 17, 17, 12, 16, 15]],
    #                [[6, 9, 9, 3, 5, 4, 9], [5, 6, 5, 6, 5, 5, 4], [15, 18, 10, 15, 16, 15, 11]],
    #                [[7, 9, 6, 8, 7, 8, 3], [5, 4, 4, 4, 5, 6, 5], [16, 13, 10, 14, 11, 13, 9]],
    #                [[9, 3, 9, 5, 7, 6, 4], [6, 6, 5, 5, 6, 6, 4], [9, 9, 18, 11, 18, 13, 12]],
    #                [[4, 8, 3, 4, 5, 3, 9], [5, 4, 5, 4, 5, 6, 4], [10, 11, 9, 14, 15, 14, 13]]
    #                ])

    JW = np.array([[1, 2, 5], [1, 3, 4], [2, 3, 5], [1, 3, 4], [2, 4, 5], [1, 4, 5], [2, 3, 5]])

    Schedule_strategy = MOEA_D1(pop_size, XOVR, MUTR, J, S, TM, JW, JmNumber, terminate, Es, u, fmax, lamda, archive)
    # Schedule_strategy.plot_figure()
    Schedule_strategy.Write_excel(155, 6)
    # Schedule_strategy.plot_fatig()

