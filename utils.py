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



# 1 聚类参考点
class NSGA_vector1:
    # ChoseN聚类点个数,points帕累托最优近似集内的点,
    # nobj目标个数,R需要生成的参考点数量（权重数量）;dc距离最大值
    def __init__(self,R,dc,nobj=3):
        self.dc = dc
        self.R = R
        self.nobj = nobj

    #计算距离
    def dist(self,x,y): #欧式距离
        return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2+(x[1]-y[1])**2)

    #计算局部密度
    def rou(self,points):
        irou=np.zeros((len(points),1),dtype=int)
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j and self.dist(points[i],points[j]) < self.dc:
                    irou[i] += 1
        return irou

    # 聚类点选择
    def ClusterPoint(self,points,irou,ChoseN): #ChoseN聚类点个数
        epsi = np.zeros((len(points),1)) #各点到比该点更高密度的点的距离
        index_max = np.argmax(irou)  #最高密度点
        CPoints = points
        for i in range(len(points)):
            if i == index_max: #最高密度点不存在该距离
                mindist = np.inf
            else:
                mindist = self.dist(points[i],points[index_max]) #先赋值
                for j in range(len(points)): #j可能取到index_max
                    if i != j and irou[j]>irou[i]: #到高密度点距离d
                        if self.dist(points[i],points[j]) < mindist: #更新距离d，需要找到最小
                            mindist = self.dist(points[i],points[j])
            epsi[i] = mindist
        index = np.argsort(-epsi[:,0]) #从大到小,返回的索引为列表
        points = points[index]
        CPoints = points[0:ChoseN]  #前ChoseN个点
        return CPoints

    #参考点生成，R：总参考点数量
    #所有数组的索引与Cpoints一致
    def genweight(self,Cpoints,Rpoints,R,ChoseN): #Cpoints、nu、tempdist对应相等
        nu = np.zeros(ChoseN,dtype=int) #每个聚集点附近点的数量
        # gn = np.zeros(ChoseN,dtype=int) #每个聚集点附近参考点的数量
        # wn = np.zeros((ChoseN,6)) #每个聚集点的边界,前3表示3个维度的间距，后3表示3个维度的最小值
        pointlist = np.zeros(len(Rpoints))#每个R对应的中心点
        for i in range(len(Rpoints)):
            tempdist = []
            for j in range(len(Cpoints)):   #找最近的聚集点
                tempdist.append(self.dist(Rpoints[i], Cpoints[j]))
            index = np.argmin(tempdist) #最小距离的中心点序号
            nu[index] += 1
            pointlist[i] = index

        #有的聚集点没有分配到参考点哎
        Cpoints_new = copy.deepcopy(Cpoints)
        #重新Rpoint、Cpoint
        for a in range(len(Cpoints)):
            if a not in pointlist:#pointlist存储的是中心点的序号#没用上的聚集点
                c = np.array([Cpoints[a]])
                Rpoints = np.append(Rpoints,c,axis=0)
                a_new=np.where(Cpoints_new==Cpoints[a])[0] #np.where可返回多个地方
                Cpoints_new = np.delete(Cpoints_new,a_new,axis=0) #delete第二个参数是索引

        nu = np.zeros(len(Cpoints_new), dtype=int)  # 每个聚集点附近点的数量
        gn = np.zeros(len(Cpoints_new), dtype=int)  # 每个聚集点附近参考点的数量
        wn = np.zeros((len(Cpoints_new), 6))  # 每个聚集点的边界,前3表示3个维度的间距，后3表示3个维度的最小值
        pointlist = np.zeros(len(Rpoints))  # 每个R对应的中心点
        for k in range(len(Rpoints)):
            tempdist = []
            for l in range(len(Cpoints_new)):  # 找最近的聚集点
                tempdist.append(self.dist(Rpoints[k], Cpoints_new[l]))
            index = np.argmin(tempdist)  # 最小距离的中心点序号
            nu[index] += 1
            pointlist[k] = index

        Sum = 0
        for i in range(len(gn)-1):
            gn[i] = int(R * nu[i]/sum(nu))
            Sum += gn[i]
        gn[len(gn)-1] = R - Sum

        for i in range(len(Cpoints_new)): #每个聚集点邻域的边界
            Rp = Rpoints[pointlist == i] #中心点序号为i的参考点坐标
            minx = min(Rp[:,0])
            maxx = max(Rp[:,0])
            miny = min(Rp[:,1])
            maxy = max(Rp[:,1])
            minz = min(Rp[:,2])
            maxz = max(Rp[:,2])
            wn[i][0] = (maxx - minx) / gn[i]
            wn[i][1] = (maxy - miny) / gn[i]
            wn[i][2] = (maxz - minz) / gn[i]
            wn[i][3] = minx
            wn[i][4] = miny
            wn[i][5] = minz

        # 生成权重向量(要和Chrom个体对应)
        weights = []
        for i in range(len(gn)):  # 对每个聚集点
            for j in range(gn[i]): #生成参考点
                weight = np.zeros(3) #权重坐标
                weight[0] = wn[i][3] + wn[i][0] * j
                weight[1] = wn[i][4] + wn[i][1] * j
                weight[2] = wn[i][5] + wn[i][2] * j
                weights.append(weight)
        weights = np.array(weights)

        # 剔除重复点
        official_weight = []
        for i in range(len(weights)):  # 对每个权重
            if len(official_weight) == 0: #只有初始时
                official_weight.append(weights[i])
            else:
                flag = True
                for j in range(len(official_weight)):
                    #目前的official_weight中是否有平行向量
                    if self.isparallel(weights[i],official_weight[j]):
                        flag = False
                if flag: #没有平行向量则放入
                    official_weight.append(weights[i])

        official_weight = np.array(weights)
        return official_weight

    def isparallel(self,w1,w2):
        if w1[0]/w2[0] == w1[1]/w2[1] == w1[2]/w2[2]:
            return True
        else:
            return False

    #计算夹角
    def angle(self,x, y):
        # 分别计算两个向量的模，计算两个向量的点积，计算夹角的cos值
        cos_ = x.dot(y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y)))
        # 求得夹角（弧度制）：
        angle_hu = np.arccos(cos_)
        return angle_hu

    #分配权重
    def assignLambda(self,target,Chrom,Lambda):
        Lambda_new = copy.deepcopy(Lambda)
        Delete = copy.deepcopy(Lambda)
        for i in range(len(Chrom)):
            point = target(Chrom[i])
            ang = []
            for j in range(len(Delete)):
                ang.append(self.angle(point,Delete[j]))
            minang = min(ang)
            index_ = ang.index(minang)
            choice = Delete[ang==minang]
            for i in range(len(Lambda)):
                if (Lambda[i] == choice).all():
                    ch = i
            # ch = np.where(Lambda==choice)
            Delete = np.delete(Delete,index_,axis=0)
            Lambda_new[i] = Lambda[ch]
        return Lambda_new

    #points 帕累托前沿点(点坐标为目标值)
    def genvector(self,points,target,Chrom,ParetoPop_len,T):
        irou = self.rou(points) #计算每个点的密度
        self.ChoseN = len(points)//5
        Cpoints = self.ClusterPoint(points, irou, self.ChoseN) #得到聚集点坐标
        Rpoints = []
        for point1 in points:
            flag = True
            for point2 in Cpoints:
                if (point1 == point2).all():
                    flag = False
            if flag:
                Rpoints.append(point1)
        Rpoints = np.array(Rpoints)
        # Rpoints = np.array([point for point in points if point not in Cpoints])
        # 求补集（周围参考点）坐标
        weights = self.genweight(Cpoints, Rpoints, self.R, self.ChoseN) #得到Lambda
        Lambda = self.assignLambda(target,Chrom,weights)

        dist = np.zeros((ParetoPop_len, ParetoPop_len))
        # lambda的邻居序号
        for i in range(ParetoPop_len - 1):
            for j in range(i + 1, ParetoPop_len):
                dist[i][j] = dist[j][i] = np.linalg.norm(Lambda[i] - Lambda[j])
        sorts = np.argsort(dist, axis=1)  # 按行升序排列,每个元素最近的T个邻居都在前面
        sp_neighbors = np.arange(ParetoPop_len)  # 0 - popsize-1
        sp_neighbors = sp_neighbors[sorts]
        # T = ParetoPop_len//4  #邻居个数取前一半
        sp_neighbors = sp_neighbors[:, :T]
        return Lambda, sp_neighbors   #告知有几个邻居

    def genvector1(self,points,target,Chrom,ParetoPop_len):
        irou = self.rou(points) #计算每个点的密度
        self.ChoseN = len(points)//5
        Cpoints = self.ClusterPoint(points, irou, self.ChoseN) #得到聚集点坐标
        Rpoints = []
        for point1 in points:
            flag = True
            for point2 in Cpoints:
                if (point1 == point2).all():
                    flag = False
            if flag:
                Rpoints.append(point1)
        Rpoints = np.array(Rpoints)
        # Rpoints = np.array([point for point in points if point not in Cpoints])
        # 求补集（周围参考点）坐标
        weights = self.genweight(Cpoints, Rpoints, self.R, self.ChoseN) #得到Lambda
        Lambda = self.assignLambda(target,Chrom,weights)

        dist = np.zeros((ParetoPop_len, ParetoPop_len))
        # lambda的邻居序号
        for i in range(ParetoPop_len - 1):
            for j in range(i + 1, ParetoPop_len):
                dist[i][j] = dist[j][i] = np.linalg.norm(Lambda[i] - Lambda[j])
        sorts = np.argsort(dist, axis=1)  # 按行升序排列,每个元素最近的T个邻居都在前面
        sp_neighbors = np.arange(ParetoPop_len)  # 0 - popsize-1
        sp_neighbors = sp_neighbors[sorts]
        return Lambda, sp_neighbors   #告知有几个邻居

class Kmeans_vector:
    # ChoseN聚类点个数,points帕累托最优近似集内的点,
    # nobj目标个数,R需要生成的参考点数量（权重数量）;dc距离最大值
    def __init__(self,R,dc,nobj=3):
        self.dc = dc
        self.R = R
        self.nobj = nobj

    # 两点距离
    def distance(self, e1, e2):
        return np.sqrt((e1[0] - e2[0]) ** 2 + (e1[1] - e2[1]) ** 2 + (e1[2] - e2[2]) ** 2)

    # 集合中心
    def means(self, arr):
        return [np.mean([e[0] for e in arr]), np.mean([e[1] for e in arr]), np.mean([e[2] for e in arr])]

    # arr中距离a最远的元素，用于初始化聚类中心
    def farthest(self, k_arr, arr):
        f = [0, 0]
        max_d = 0
        for e in arr:
            d = 0
            for i in range(k_arr.__len__()):
                d = d + np.sqrt(self.distance(k_arr[i], e))
            if d > max_d:
                max_d = d
                f = e
        return f

    # arr中距离a最近的元素，用于聚类
    def closest(self, a, arr):
        c = arr[1]
        min_d = self.distance(a, arr[1])
        arr = arr[1:]
        for e in arr:
            d = self.distance(a, e)
            if d < min_d:
                min_d = d
                c = e
        return c


    # fig = plt.figure(1)
    # ax = fig.gca(projection='3d')
    # ## 生成二维随机坐标，手上有数据集的朋友注意，理解arr改起来就很容易了
    # ## arr是一个数组，每个元素都是一个二元组，代表着一个坐标
    # ## arr形如：[ (x1, y1), (x2, y2), (x3, y3) ... ]
    # arr = np.random.uniform(low=0.0, high=100.0, size=(100, 3))  # 100个点，每个点横纵坐标范围为100
    # ## 初始化聚类中心和聚类容器
    # m = 5  # 5个聚类中心
    # r = np.random.randint(arr.__len__() - 1)
    # k_arr = np.array([arr[r]])
    # cla_arr = [[]]
    # for i in range(m - 1):
    #     k = farthest(k_arr, arr)
    #     k_arr = np.concatenate([k_arr, np.array([k])])
    #     cla_arr.append([])
    #
    # ## 迭代聚类
    # n = 20
    # cla_temp = cla_arr
    # for i in range(n):  # 迭代n次
    #     for e in arr:  # 把集合里每一个元素聚到最近的类
    #         ki = 0  # 假定距离第一个中心最近
    #         min_d = distance(e, k_arr[ki])
    #         for j in range(1, k_arr.__len__()):
    #             if distance(e, k_arr[j]) < min_d:  # 找到更近的聚类中心
    #                 min_d = distance(e, k_arr[j])
    #                 ki = j
    #         cla_temp[ki].append(e)
    #     # 迭代更新聚类中心
    #     for k in range(k_arr.__len__()):
    #         if n - 1 == i:
    #             break
    #         k_arr[k] = means(cla_temp[k])
    #         cla_temp[k] = []
    #
    # ## 可视化展示
    # col = ['HotPink', 'Aqua', 'Chartreuse', 'yellow', 'LightSalmon']
    # for i in range(m):
    #     ax.scatter(k_arr[i][0], k_arr[i][1], k_arr[i][2], linewidth=10, color=col[i])
    #     ax.scatter([e[0] for e in cla_temp[i]], [e[1] for e in cla_temp[i]], [e[2] for e in cla_temp[i]], color=col[i])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()

    #计算距离
    def dist(self,x,y): #欧式距离
        return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2+(x[1]-y[1])**2)

    #计算局部密度
    def rou(self,points):
        irou=np.zeros((len(points),1),dtype=int)
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j and self.dist(points[i],points[j]) < self.dc:
                    irou[i] += 1
        return irou

    # 聚类点选择
    def ClusterPoint(self,points,irou,ChoseN): #ChoseN聚类点个数
        epsi = np.zeros((len(points),1)) #各点到比该点更高密度的点的距离
        index_max = np.argmax(irou)  #最高密度点
        CPoints = points
        for i in range(len(points)):
            if i == index_max: #最高密度点不存在该距离
                mindist = np.inf
            else:
                mindist = self.dist(points[i],points[index_max]) #先赋值
                for j in range(len(points)): #j可能取到index_max
                    if i != j and irou[j]>irou[i]: #到高密度点距离d
                        if self.dist(points[i],points[j]) < mindist: #更新距离d，需要找到最小
                            mindist = self.dist(points[i],points[j])
            epsi[i] = mindist
        index = np.argsort(-epsi[:,0]) #从大到小,返回的索引为列表
        points = points[index]
        CPoints = points[0:ChoseN]  #前ChoseN个点
        return CPoints

    #参考点生成，R：总参考点数量
    #所有数组的索引与Cpoints一致
    def genweight(self,Cpoints,Rpoints,R,ChoseN): #Cpoints、nu、tempdist对应相等
        # R = 50, ChoseN = 10
        nu = np.zeros(ChoseN,dtype=int) #每个聚集点附近点的数量
        pointlist = np.zeros(len(Rpoints))#每个R对应的中心点
        for i in range(len(Rpoints)):
            tempdist = []
            for j in range(len(Cpoints)):   #找最近的聚集点
                tempdist.append(self.dist(Rpoints[i], Cpoints[j]))
            index = np.argmin(tempdist) #最小距离的中心点序号
            nu[index] += 1
            pointlist[i] = index

        print(pointlist)
        #有的聚集点没有分配到参考点哎
        Cpoints_new = copy.deepcopy(Cpoints)
        #重新Rpoint、Cpoint
        for a in range(len(Cpoints)):
            if a not in pointlist:#pointlist存储的是中心点的序号#没用上的聚集点
                c = np.array([Cpoints[a]])
                Rpoints = np.append(Rpoints,c,axis=0)
                a_new=np.where(Cpoints_new==Cpoints[a])[0] #np.where可返回多个地方
                Cpoints_new = np.delete(Cpoints_new,a_new,axis=0) #delete第二个参数是索引


        nu = np.zeros(len(Cpoints_new), dtype=int)  # 每个聚集点附近点的数量
        gn = np.zeros(len(Cpoints_new), dtype=int)  # 每个聚集点附近参考点的数量
        pointlist = np.zeros(len(Rpoints))  # 每个R对应的中心点
        for k in range(len(Rpoints)):
            tempdist = []
            for l in range(len(Cpoints_new)):  # 找最近的聚集点
                tempdist.append(self.dist(Rpoints[k], Cpoints_new[l]))
            index = np.argmin(tempdist)  # 最小距离的中心点序号
            nu[index] += 1
            pointlist[k] = index
        print(pointlist)
        Sum = 0
        for i in range(len(gn)-1):
            gn[i] = int(R * nu[i]/sum(nu))
            Sum += gn[i]
        gn[len(gn)-1] = R - Sum
        print(pointlist)
        print(len(pointlist))
        official_weight = []
        # 生成权重向量(要和Chrom个体对应)
        for i in range(len(Cpoints_new)): #每个聚集点邻域的边界
            Rp = Rpoints[pointlist == i] #中心点序号为i的参考点坐标
            weights = np.zeros((gn[i], 3))
            p2 = Rp[np.argmax(Rp[:,0])]
            p4 = Rp[np.argmax(Rp[:,1])]
            p6 = Rp[np.argmax(Rp[:,2])]

            fig = plt.figure(1)
            ax = fig.gca(projection='3d')
            x1, y1, z1 = p2[0],p2[1],p2[2]
            x3, y3, z3 = p4[0],p4[1],p4[2]
            x2, y2, z2 = p6[0],p6[1],p6[2]
            sample_size = gn[i]
            theta = np.arange(0, 1, 0.001)
            x = theta * x1 + (1 - theta) * x2
            y = theta * y1 + (1 - theta) * y2
            z = theta * z1 + (1 - theta) * z2
            ax.plot(x, y, z, 'g--', linewidth=2)
            x = theta * x1 + (1 - theta) * x3
            y = theta * y1 + (1 - theta) * y3
            z = theta * z1 + (1 - theta) * z3
            ax.plot(x, y, z, 'g--', linewidth=2)
            x = theta * x3 + (1 - theta) * x2
            y = theta * y3 + (1 - theta) * y2
            z = theta * z3 + (1 - theta) * z2
            ax.plot(x, y, z, 'g--', linewidth=2)
            rnd1 = np.random.random(size=sample_size)
            rnd2 = np.random.random(size=sample_size)
            rnd2 = np.sqrt(rnd2)
            weights[:, 0] = rnd2 * (rnd1 * x1 + (1 - rnd1) * x2) + (1 - rnd2) * x3
            weights[:, 1] = rnd2 * (rnd1 * y1 + (1 - rnd1) * y2) + (1 - rnd2) * y3
            weights[:, 2] = rnd2 * (rnd1 * z1 + (1 - rnd1) * z2) + (1 - rnd2) * z3
            official_weight.append(weights.tolist())

        print(official_weight)
        official_weight = official_weight[0]
        print(official_weight)
        official_weight = np.array(official_weight)
        print("1")
        print(official_weight)
        print(np.shape(official_weight))
        return official_weight

    def isparallel(self,w1,w2):
        if w1[0]/w2[0] == w1[1]/w2[1] == w1[2]/w2[2]:
            return True
        else:
            return False

    #计算夹角
    def angle(self,x, y):
        # 分别计算两个向量的模，计算两个向量的点积，计算夹角的cos值
        cos_ = x.dot(y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y)))
        # 求得夹角（弧度制）：
        angle_hu = np.arccos(cos_)
        return angle_hu

    #分配权重
    def assignLambda(self,target,Chrom,Lambda):
        Lambda_new = copy.deepcopy(Lambda)
        Delete = copy.deepcopy(Lambda)
        print(Delete)
        for i in range(len(Chrom)):
            point = target(Chrom[i])
            ang = []
            for j in range(len(Delete)):
                ang.append(self.angle(point,Delete[j]))
            minang = min(ang)
            index_ = ang.index(minang)
            choice = Delete[ang==minang]
            for i in range(len(Lambda)):
                if (Lambda[i] == choice).all():
                    ch = i
            # ch = np.where(Lambda==choice)
            Delete = np.delete(Delete,index_,axis=0)
            Lambda_new[i] = Lambda[ch]
        return Lambda_new

    #points 帕累托前沿点(点坐标为目标值)
    def genvector(self,points,target,Chrom,ParetoPop_len,T):
        irou = self.rou(points) #计算每个点的密度
        self.ChoseN = len(points)//5
        Cpoints = self.ClusterPoint(points, irou, self.ChoseN) #得到聚集点坐标
        Rpoints = []
        for point1 in points:
            flag = True
            for point2 in Cpoints:
                if (point1 == point2).all():
                    flag = False
            if flag:
                Rpoints.append(point1)
        Rpoints = np.array(Rpoints)
        # Rpoints = np.array([point for point in points if point not in Cpoints])
        # 求补集（周围参考点）坐标
        weights = self.genweight(Cpoints, Rpoints, self.R, self.ChoseN) #得到Lambda
        Lambda = self.assignLambda(target,Chrom,weights)

        dist = np.zeros((ParetoPop_len, ParetoPop_len))
        # lambda的邻居序号
        for i in range(ParetoPop_len - 1):
            for j in range(i + 1, ParetoPop_len):
                dist[i][j] = dist[j][i] = np.linalg.norm(Lambda[i] - Lambda[j])
        sorts = np.argsort(dist, axis=1)  # 按行升序排列,每个元素最近的T个邻居都在前面
        sp_neighbors = np.arange(ParetoPop_len)  # 0 - popsize-1
        sp_neighbors = sp_neighbors[sorts]
        # T = ParetoPop_len//4  #邻居个数取前一半
        sp_neighbors = sp_neighbors[:, :T]
        return Lambda, sp_neighbors   #告知有几个邻居

    def genvector1(self,points,target,Chrom,ParetoPop_len):
        irou = self.rou(points) #计算每个点的密度
        self.ChoseN = len(points)//5
        Cpoints = self.ClusterPoint(points, irou, self.ChoseN) #得到聚集点坐标
        Rpoints = []
        for point1 in points:
            flag = True
            for point2 in Cpoints:
                if (point1 == point2).all():
                    flag = False
            if flag:
                Rpoints.append(point1)
        Rpoints = np.array(Rpoints)
        # Rpoints = np.array([point for point in points if point not in Cpoints])
        # 求补集（周围参考点）坐标
        weights = self.genweight(Cpoints, Rpoints, self.R, self.ChoseN) #得到Lambda
        Lambda = self.assignLambda(target,Chrom,weights)

        dist = np.zeros((ParetoPop_len, ParetoPop_len))
        # lambda的邻居序号
        for i in range(ParetoPop_len - 1):
            for j in range(i + 1, ParetoPop_len):
                dist[i][j] = dist[j][i] = np.linalg.norm(Lambda[i] - Lambda[j])
        sorts = np.argsort(dist, axis=1)  # 按行升序排列,每个元素最近的T个邻居都在前面
        sp_neighbors = np.arange(ParetoPop_len)  # 0 - popsize-1
        sp_neighbors = sp_neighbors[sorts]
        return Lambda, sp_neighbors   #告知有几个邻居

class NSGA_vector:
    def __init__(self,nobj=3):
        self.nobj = nobj

    #参考点生成
    def genweight(self,ClassNum,points,ChoseN,popsize):
        #ChoseN：有几类；points是帕累托最优面上的点;ClassNum,points对应点和类别；R：需要生成的向量个数
        nu = np.zeros(ChoseN,dtype=int) #每一类points点的数量
        gn = np.zeros(ChoseN,dtype=int) #每一类需要生成参考点的数量
        wn = np.zeros((ChoseN,6)) #每一类边界,前3表示3个维度的间距，后3表示3个维度的最小值

        for i in range(ChoseN): #nu赋值，每一类有多少个点
            nu[i] = len(ClassNum[ClassNum == i])
        for i in range(ChoseN): #gn赋值，每一类需要生成多少个点
            gn[i] = int(popsize * nu[i]/sum(nu))

        print('ClassNum',ClassNum)
        print('points', points)
        print('ChoseN',ChoseN)
        for i in range(ChoseN): #每类边界
            Rp = points[ClassNum == i] #所有第i类的点的坐标
            minx = min(Rp[:,0])
            maxx = max(Rp[:,0])
            miny = min(Rp[:,1])
            maxy = max(Rp[:,1])
            minz = min(Rp[:,2])
            maxz = max(Rp[:,2])
            wn[i][0] = (maxx - minx) / gn[i]
            wn[i][1] = (maxy - miny) / gn[i]
            wn[i][2] = (maxz - minz) / gn[i]
            wn[i][3] = minx
            wn[i][4] = miny
            wn[i][5] = minz

        # 生成权重向量(要和Chrom个体对应)
        weights = []
        for i in range(ChoseN):  # 对每一类
            for j in range(gn[i]): #生成参考点
                weight = np.zeros(3) #权重坐标
                # weight[0] = wn[i][3] + wn[i][0] * j
                # weight[1] = wn[i][4] + wn[i][1] * j
                # weight[2] = wn[i][5] + wn[i][2] * j

                weights.append(weight)
        weights = np.array(weights)
        return weights

    def angle(self,x, y):
        cos_ = x.dot(y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y)))
        # 求得夹角（弧度制）
        angle_hu = np.arccos(cos_)
        return angle_hu

    #分配权重
    def assignLambda(self,Popfun,Lambda):
        Lambda_new = copy.deepcopy(Lambda)
        _Lambda = copy.deepcopy(Lambda)
        for i in range(len(Popfun)):
            ang = []
            for j in range(len(_Lambda)):
                ang.append(self.angle(Popfun[i],_Lambda[j]))
            index_ = np.argmin(ang)
            Lambda_new[i] = _Lambda[index_]
            _Lambda = np.delete(_Lambda,index_,axis=0)
        return Lambda_new

    #points 帕累托前沿点(点坐标为目标值)
    def genvector(self,points,Popfun,popsize,ChoseN,T):
        km = Kmeans(k=ChoseN)
        ClassNum = km.predict(points) #得到每个点的分类号
        Lambda = self.genweight(ClassNum,points,ChoseN,popsize) #得到Lambda
        Lambda = self.assignLambda(Popfun,Lambda) #分配权重

        dist = np.zeros((popsize, popsize))
        # lambda的邻居序号
        for i in range(popsize-1):
            for j in range(i + 1, popsize):
                dist[i][j] = dist[j][i] = np.sqrt(np.power((Lambda[i] - Lambda[j]),2).sum())
        sorts = np.argsort(dist, axis=1)  # 按行升序排列,每个元素最近的T个邻居都在前面
        sp_neighbors = np.arange(popsize)  # 0 - popsize-1
        sp_neighbors = sp_neighbors[sorts]
        sp_neighbors = sp_neighbors[:, :T]
        return Lambda, sp_neighbors

class Kmeans():
    # K:聚类数目；Max_iter:最大迭代次数，Var：判断是否收敛
    def __init__(self, k, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon

    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # 返回距离该样本最近的一个中心索引[0, self.k)
    def _closest_centroid(self, sample, centroids):
        distances = self.euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)
        return closest_i

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)  # 记录每个聚类中心周围的点（序号）
        return clusters

    # 对中心进行更新
    def update_centroids(self, clusters, X):  # 聚类中心序号是1和5：[[1,2,3,4],[5,6,7,8]]；X：所有点坐标
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)  # 每一堆的中心点在哪里
            centroids[i] = centroid  # 更新这一堆的中心点
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
        return centroids

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):  # 中心点是什么不重要，重要的是根据中心点将所有的个体分好类
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i  # 类的编号（第一类、第二类..）
        return y_pred

    # 正规化数据集 X
    def normalize(self, X, axis=-1, p=2):
        lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
        lp_norm[lp_norm == 0] = 1
        return X / np.expand_dims(lp_norm, axis)

    # 计算一个样本与数据集中所有样本的欧氏距离的平方
    def euclidean_distance(self, one_sample, X):
        one_sample = one_sample.reshape(1, -1)
        X = X.reshape(X.shape[0], -1)
        distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
        return distances

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X):  # X是输入的三维点
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(X)
        print('centroids',centroids)
        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        for _ in range(self.max_iterations):
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, X)  # 聚类中心序号是1和5：[[1,2,3,4],[5,6,7,8]]
            print('clusters',clusters)
            former_centroids = centroids  # 之前的聚类点

            # 计算新的聚类中心，对所有样本重新分类
            for i in range(len(clusters)):
                if len(clusters[i])==0:
                    centroids[i] = X[np.random.choice(range(len(X)))]
            centroids = self.update_centroids(clusters, X)
            print('centroids', centroids)

            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            # 中心点很可能不是样本点
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break
        return self.get_cluster_labels(clusters, X)

# 2 评价指标
class Estimate():
    def IGD(self,popfun, PF):
        distance = np.min(self.EuclideanDistances(PF, popfun), 1)
        score = np.mean(distance)
        return score

    def EuclideanDistances(self,A, B):
        BT = B.transpose()
        vecProd = np.dot(A, BT)
        SqA = A ** 2
        sumSqA = np.matrix(np.sum(SqA, axis=1))
        sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

        SqB = B ** 2
        sumSqB = np.sum(SqB, axis=1)
        sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
        SqED = sumSqBEx + sumSqAEx - 2 * vecProd
        SqED[SqED < 0] = 0.0
        ED = np.sqrt(SqED)
        return ED

    def Coverage(self,A, B):
        Sel = Select()
        numB = 0  # B中被支配的个体
        for i in B:  # 对于集合B中的每一个个体
            for j in A:
                if Sel.isDominates(j, i):  # 至少被A中一个解支配
                    numB += 1
                    break
        ratio = numB / len(B)
        return ratio

# 3 聚合函数
class AggFun():
    def TchebycheffFunc(self,target, Chrom, y, Bi, z, Lambda):  # Bi是idx的邻居序号列表
        # 若gy<gx更新,用的权重是邻居的权重
        fy = target(y)
        for j in range(len(Bi)):
            w = Lambda[Bi[j]]  # 个体的权重
            maxn_y = max(w * abs(fy - z))  # y的适应度与最优之差乘以权重，最大的那个目标值之差
            maxn_x = max(w * abs(target(Chrom[Bi[j]]) - z))
            # idx某个邻居的适应度与最优之差乘以权重，最大的那个目标值之差
            if maxn_x >= maxn_y:  # 与最优目标值的最大差值，小一点好
                Chrom[Bi[j]] = y  # 个体idx的这个邻居(也就是种群的某个个体)被y代替

    def Tcheb_sum(self,target, Chrom, y, Bi, z, Lambda):  # Bi是idx的邻居序号列表
        # 若gy<gx更新,用的权重是邻居的权重
        rou = 0.5
        fy = target(y)
        for j in range(len(Bi)):
            w = Lambda[Bi[j]]  # 个体的权重
            maxn_y = max(w * abs(fy - z))  # y的适应度与最优之差乘以权重，最大的那个目标值之差
            maxn_x = max(w * abs(target(Chrom[Bi[j]]) - z))
            g_y = maxn_y + rou * sum(abs(fy - z))
            g_x = maxn_x + rou * sum(abs(target(Chrom[Bi[j]]) - z))
            if g_x >= g_y:  # 与最优目标值的最大差值，小一点好
                Chrom[Bi[j]] = y  # 个体idx的这个邻居(也就是种群的某个个体)被y代替

    #切比雪夫简化时间版
    def TchebycheffFunc1(self,target,target_pop, Pop, y, Bi, z, Lambda):  # Bi是idx的邻居序号列表
        fy = target(y)
        pop_fun = target_pop(Pop)
        # 若gy<gx更新,用的权重是邻居的权重
        for j in range(len(Bi)):
            w = Lambda[Bi[j]]  # 个体的权重
            maxn_y = max(w * abs(fy - z))  # y的适应度与最优之差乘以权重，最大的那个目标值之差
            maxn_x = max(w * abs(pop_fun[Bi[j]] - z))
            if maxn_x >= maxn_y:
                Pop[Bi[j]] = y
                pop_fun[Bi[j]] = fy

    #切比雪夫加权简化时间版
    def Tcheb_sum1(self,target, target_pop,Pop, y, Bi, z, Lambda):  # Bi是idx的邻居序号列表
        # 若gy<gx更新,用的权重是邻居的权重
        rou = 0.5
        fy = target(y)
        popfun = target_pop(Pop)
        for j in range(len(Bi)):
            w = Lambda[Bi[j]]  # 个体的权重
            maxn_y = max(w * abs(fy - z))  # y的适应度与最优之差乘以权重，最大的那个目标值之差
            maxn_x = max(w * abs(popfun[Bi[j]] - z))
            g_y = maxn_y + rou * sum(abs(fy - z))
            g_x = maxn_x + rou * sum(abs(popfun[Bi[j]] - z))
            if g_x >= g_y:  # 与最优目标值的最大差值，小一点好
                Pop[Bi[j]] = y  # 个体idx的这个邻居(也就是种群的某个个体)被y代替

    #聚合进化正常版
    def isdominates_all(self, s, EPs,target,archive):
        Sel = Select()
        fun_s = target(s)
        ep = True  # 决定y是否放进EPs,True就放
        delete = []  # 装EPs个体
        for EP in EPs:  # EPs是所有支配个体，EPs外部存档
            fun_EP = target(EP)
            if (fun_s == fun_EP).all():  # 目标值完全相等的情况
                ep = False
                break  # 跳出当前的for
            elif Sel.isDominates(fun_s, fun_EP):  # y支配EPs[k]
                delete.append(EP)  # 准备删除EPs[k]
            elif Sel.isDominates(fun_EP, fun_s):  # EPs[k]支配y
                ep = False
            else:  # 不存在支配关系也不相等
                pass  # 默认可能放可能不放
        if len(delete) != 0:
            for k in range(len(delete)):
                EPs.remove(delete[k])
        if ep == True:
            EPs.append(list(s))  # EPs[k]被支配了不配在EP集合当中了，y能支配EPs[k]，放进来
        while len(EPs) > archive:  # 存档数目
            select = np.random.randint(0, len(EPs))
            del EPs[select]  # 超过存档数目了就随机剔除

    #变邻域时的时间简化版正常版
    def isdominates_all1(self,s,EPs,target,fun_EPs,archive):
        Sel = Select()
        ep = True  # 决定y是否放进EPs,True就放
        delete = []  # 装EPs个体
        delete_ = []
        fun_s = list(target(s))
        for i in range(len(EPs)):  # EPs是所有支配个体，EPs外部存档
            if (np.array(fun_s) == np.array(fun_EPs[i])).all():  # 目标值完全相等的情况
                ep = False
                break  # 跳出当前的for
            elif Sel.isDominates(np.array(fun_s), np.array(fun_EPs[i])):  # y支配EPs[k]
                delete.append(EPs[i])  # 准备删除EPs[k]
                delete_.append(fun_EPs[i])
            elif Sel.isDominates(np.array(fun_EPs[i]), np.array(fun_s)):  # EPs[k]支配y
                ep = False
                break
            else:  # 不存在支配关系也不相等
                pass  # 默认放
        if len(delete) != 0:
            for k in range(len(delete)):
                EPs.remove(delete[k])
                fun_EPs.remove(delete_[k])
        if ep == True:
            EPs.append(list(s))  # EPs[k]被支配了不配在EP集合当中了，y能支配EPs[k]，放进来
            fun_EPs.append(fun_s)
        while len(EPs) > archive:  # 存档数目
            select = np.random.randint(0, len(EPs))
            del EPs[select]  # 超过存档数目了就随机剔除
            del fun_EPs[select]

    #聚合进化不正常版
    def isdominates_all2(self, s, EPs,target,archive):
        Sel = Select()
        fun_s = target(s)
        for EP in EPs:  # EPs是所有支配个体，EPs外部存档
            fun_EP = target(EP)
            if (fun_s == fun_EP).all():  # 目标值完全相等的情况
                break  # 跳出当前的for
            elif Sel.isDominates(fun_s, fun_EP):  # y支配EPs[k]
                if EP in EPs:
                    EPs.remove(EP)
                EPs.append(list(s))  # EPs[k]被支配了不配在EP集合当中了，y能支配EPs[k]，放进来
            elif Sel.isDominates(fun_EP, fun_s):  # EPs[k]支配y
                break
            else:  # 不存在支配关系也不相等
                EPs.append(list(s))  # EPs[k]被支配了不配在EP集合当中了，y能支配EPs[k]，放进来
        while len(EPs) > archive:  # 存档数目
            select = np.random.randint(0, len(EPs))
            del EPs[select]  # 超过存档数目了就随机剔除

    #变邻域不正常版
    def isdominates_all3(self, s, EPs, target, fun_EPs, archive):
        EPs_copy = copy.deepcopy(EPs)
        fun_EPs_copy = copy.deepcopy(fun_EPs)
        Sel = Select()
        ep = True  # 决定y是否放进EPs,True就放
        fun_s = list(target(s))
        for i in range(len(EPs)):  # EPs是所有支配个体，EPs外部存档
            if (np.array(fun_s) == np.array(fun_EPs[i])).all():  # 目标值完全相等的情况
                break  # 跳出当前的for
            elif Sel.isDominates(np.array(fun_s), np.array(fun_EPs[i])):  # y支配EPs[k]
                if EPs[i] in EPs_copy:
                    EPs_copy.remove(EPs[i])
                    fun_EPs_copy.remove(fun_EPs[i])
                EPs_copy.append(list(s))  # EPs[k]被支配了不配在EP集合当中了，y能支配EPs[k]，放进来
                fun_EPs_copy.append(fun_s)
            elif Sel.isDominates(np.array(fun_EPs[i]), np.array(fun_s)):  # EPs[k]支配y
                break
            else:  # 不存在支配关系也不相等
                EPs_copy.append(list(s))  # EPs[k]被支配了不配在EP集合当中了，y能支配EPs[k]，放进来
                fun_EPs_copy.append(fun_s)
        while len(EPs) > archive:  # 存档数目
            select = np.random.randint(0, len(EPs))
            del EPs[select]  # 超过存档数目了就随机剔除
            del fun_EPs[select]
        return EPs_copy,fun_EPs_copy

# 4 非支配排序
class Select():
    # NSGA3排序
    def envselect(self,mixpop, weights, Zmin):
        # 非支配排序
        mixpopfun = self.target_pop(mixpop)
        frontno, maxfno = self.NDsort(mixpopfun)  # 非支配排序,输出每个个体的等级与最高等级
        Next = frontno < maxfno  # 不是最后一个front的位置
        # 选择最后一个front的解
        Last = np.ravel(np.array(np.where(frontno == maxfno)))  # 压扁成一维数组
        choose = self.lastselection(mixpopfun[Next, :], mixpopfun[Last, :], self.pop_size - np.sum(Next), weights, Zmin)
        Next[Last[choose]] = True
        # 生成下一代
        pop = copy.deepcopy(mixpop[Next, :])
        return pop

    def NDsort(self,mixpopfun): #按照所需数量排
        nsort = self.pop_size  # 排序个数（就是父代种群个数）
        N_large = mixpopfun.shape[0]
        # mixpop其实是输入的mixpopfun
        Loc1 = np.lexsort(mixpopfun[:, ::-1].T)  # loc1为新矩阵元素在旧矩阵中的位置，从第一列依次进行排序
        mixpop2 = mixpopfun[Loc1]
        Loc2 = Loc1.argsort()  # loc2为旧矩阵元素在新矩阵中的位置
        frontno = np.ones(N_large) * (np.inf)  # 初始化所有等级为np.inf
        # frontno[0]=1#第一个元素一定是非支配的
        maxfno = 0  # 最高等级初始化为0
        # 如果i个体被其他个体支配，则暂时不赋值目前等级
        while (np.sum(frontno < np.inf) < min(nsort, N_large)):  # 被赋予等级的个体数目不超过要排序的个体数目
            maxfno = maxfno + 1  # 最开始最高等级是1
            # 遍历每个个体，看是否属于当前等级
            for i in range(N_large):
                if (frontno[i] == np.inf):  # i还没有被赋予等级
                    dominated = 0
                    for j in range(i):  # 遍历该个体前面的个体
                        if (frontno[j] == maxfno):  # 找到等于目前等级的个体
                            m = 0  # 维度指针
                            flag = 0
                            # 在第m个目标值上i个体是劣势
                            while (m < self.nObj and mixpop2[i, m] >= mixpop2[j, m]):
                                if (mixpop2[i, m] == mixpop2[j, m]):  # 相同的个体不构成支配关系
                                    flag = flag + 1
                                m = m + 1
                            # m>=M 说明while循环是到底了的，则遍历每一维度，i个体不如j（大于等于），而且i和j不是两个完全相同的个体
                            if (m >= self.nObj and flag < self.nObj):
                                dominated = 1  # i是被支配的
                                break
                    if dominated == 0:  # 如果i不是被支配的，那i就是最高等级
                        frontno[i] = maxfno
        frontno = frontno[Loc2]  # 所有个体的等级按旧矩阵来排
        return frontno, maxfno  # 数字越小，非支配程度越高，越优

    def lastselection(self,popfun1, popfun2, K, weights, Zmin):
        # 选择最后一个front的解
        popfun = copy.deepcopy(np.vstack((popfun1, popfun2))) - np.tile(Zmin, (popfun1.shape[0] + popfun2.shape[0], 1))  # 所有点相对位置
        N, M = popfun.shape[0], popfun.shape[1]
        N1 = popfun1.shape[0]
        N2 = popfun2.shape[0]
        NZ = weights.shape[0]

        # 正则化
        extreme = np.zeros(M)  # M是数据维度 [0,0,0]
        w = np.zeros((M, M)) + 1e-6 + np.eye(M)  # [[1+1e-6,1e-6,1e-6],[1e-6,1+1e-6,1e-6],[1e-6,1e-6,1+1e-6]]
        for i in range(M):  # 对每个维度作运算
            extreme[i] = np.argmin(np.max(popfun / (np.tile(w[i, :], (N, 1))), 1))  # 找每一行最大的
        # extreme[i]存储的是第i个维度上极值点序号（在种群中的序号）

        # 计算截距
        extreme = extreme.astype(int)  # python中数据类型转换一定要用astype
        # temp = np.mat(popfun[extreme,:]).I
        temp = np.linalg.pinv(np.mat(popfun[extreme, :]))  # 把三个极值点转化为矩阵，求伪逆
        hyprtplane = np.array(np.dot(temp, np.ones((M, 1))))
        a = 1 / hyprtplane
        if np.sum(a == math.nan) != 0:  # math.nan浮点数,a浮点数个数不是0
            a = np.max(popfun, 0)  # a是每个目标的最大值
        np.array(a).reshape(M, 1)  # 一维数组转二维数组
        # a = a.T - Zmin
        a = a.T  # a存储目标最大值[10,10,10]
        popfun = popfun / (np.tile(a, (N, 1)))  # N个截距

        ##联系每一个解和对应向量
        # 计算每一个解最近的参考线的距离
        cos = self.pdist(popfun, weights)  # 混合种群每个个体和权重的夹角
        distance = np.tile(np.array(np.sqrt(np.sum(popfun ** 2, 1))).reshape(N, 1), (1, NZ)) * np.sqrt(1 - cos ** 2)
        # 联系每一个解和对应的向量
        d = np.min(distance.T, 0)  # distance.T:shape:[NZ,N]
        pi = np.argmin(distance.T, 0)  # 每个向量序号

        # 计算z关联的个数
        rho = np.zeros(NZ)  # 需要NZ个个体，同时NZ也是父代的个数Z，也是向量个数
        for i in range(NZ):
            rho[i] = np.sum(pi[:N1] == i)  # 前N1个个体中对应向量i的有多少个

        # 选出剩余的K个
        choose = np.zeros(N2)
        choose = choose.astype(bool)
        zchoose = np.ones(NZ)
        zchoose = zchoose.astype(bool)
        while np.sum(choose) < K:
            # 选择最不拥挤的参考点
            temp = np.ravel(np.array(np.where(zchoose == True)))
            jmin = np.ravel(np.array(np.where(rho[temp] == np.min(rho[temp]))))  # 最不拥挤的成为一个集合
            j = temp[jmin[np.random.randint(jmin.shape[0])]]
            #        I = np.ravel(np.array(np.where(choose == False)))
            #        I = np.ravel(np.array(np.where(pi[(I+N1)] == j)))
            I = np.ravel(np.array(np.where(pi[N1:] == j)))
            I = I[choose[I] == False]
            if (I.shape[0] != 0):
                if (rho[j] == 0):
                    s = np.argmin(d[N1 + I])
                else:
                    s = np.random.randint(I.shape[0])
                choose[I[s]] = True
                rho[j] = rho[j] + 1
            else:
                zchoose[j] = False
        return choose

    def pdist(self,x, y):
        x0 = x.shape[0]
        y0 = y.shape[0]
        xmy = np.dot(x, y.T)  # x乘以y
        xm = np.array(np.sqrt(np.sum(x ** 2, 1))).reshape(x0, 1)  # (np.sum(x**2,1):x每一行求和
        ym = np.array(np.sqrt(np.sum(y ** 2, 1))).reshape(1, y0)
        xmmym = np.dot(xm, ym)
        cos = xmy / xmmym
        return cos

    #NSGA2 排序
    def nonDominationSort(self, PopFun):
        nPop,nF = PopFun.shape[0],PopFun.shape[1]
        ranks = np.zeros(nPop, dtype=np.int32)
        nPs = np.zeros(nPop)  # 每个个体p被支配解的个数
        sPs = []  # 每个个体支配的解的集合，把索引放进去
        for i in range(nPop):
            iSet = []  # 解i的支配解集
            for j in range(nPop):
                if i == j:
                    continue
                isDom1 = PopFun[i] <= PopFun[j]
                isDom2 = PopFun[i] < PopFun[j]
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

    def roulette(self,select_list):
        sum_val = sum(select_list)
        random_val = random.random()
        probability = 0  # 累计概率
        for i in range(len(select_list)):
            probability += select_list[i] / sum_val  # 加上该个体的选中概率
            if probability >= random_val:
                return i  # 返回被选中的下标
            else:
                continue

# 5 初始化方式
class Initial():
    def prioc_MS(self,OS,decode_OS,W,JmNumber):#相同工序优先优先原则
        a,b = decode_OS(OS) # a是工序，b是工件
        MS = np.zeros(W, dtype=int)
        temp = np.zeros(JmNumber,dtype=int) #每个机器上一次加工哪道工序
        for i in range(W):
            if a[i] == 1:
                m_ = np.array(np.where(temp==0))[0]  #那些还没加工过的机器集合
                if len(m_)==0:
                    MS[i] = random.randint(1,JmNumber)  ##随机选择一个机器
                else:
                    MS[i] = random.choice(m_) + 1  ##随机选择一个机器
            elif len(np.where(temp == a[i])[0]) == 0:
                MS[i] = random.randint(1, 7)  ##随机选择一个机器
            else:
                if random.random() < 0.5:
                    MS[i] = random.choice(np.where(temp == a[i])[0]) + 1
                else:
                    MS[i] = random.randint(1, 7)  ##随机选择一个机器
            temp[MS[i] - 1] = OS[i]
        return MS

    def prio_MS(self,OS,decode_OS,W,TM):  # 加工时间短优先原则
        a, b = decode_OS(OS)  # a是工序，b是工件
        MS = np.zeros(W, dtype=int)
        for i in range(W):
            MS[i]=self.roulette(TM[b[i] - 1][a[i] - 1])+1
        return MS

    def rand_MS(self,W,JmNumber):
        # 机器序列MS,人员序列WS
        MS = np.zeros(W, dtype=int)
        for i in range(W):
            MS[i] = random.randint(1, JmNumber)  ##随机选择一个机器
        return MS

    def roulette(self,select_list):
        sum_val = sum(select_list)
        random_val = random.random()
        probability = 0  # 累计概率
        for i in range(len(select_list)):
            probability += select_list[i] / sum_val  # 加上该个体的选中概率
            if probability >= random_val:
                return i  # 返回被选中的下标
            else:
                continue

    def prio_WS(self,MS,W,JW,u):  # 加工时间短优先原则
        WS = np.zeros(W, dtype=int)
        for i in range(W):
            tmp = JW[MS[i] - 1] - 1  ##机器对应的可选人员
            WS[i] = self.roulette(u[tmp]) + 1
        return WS

    def rand_WS(self,MS,W,JW):
        WS = np.zeros(W, dtype=int)
        for i in range(W):
            tmp = JW[MS[i] - 1]  ##机器对应的可选人员
            WS[i] = np.random.choice(tmp)  ##随机选择一个人员
        return WS

# 6 进化算子
class Evolution():
    def strategy1_OS(self, s1, s2,J,W,S):
        parentspring1 = copy.deepcopy(s1)
        parentspring2 = copy.deepcopy(s2)
        temp = np.zeros(J, dtype=int)  # 子代中每个工件已经有几道工序了
        offspring = np.zeros(3 * W, dtype=int)
        at1 = 0  # parent1指针
        at2 = 0  # parent2指针
        at = True  # 从哪个parent复制
        for i in range(len(offspring)//3):
            while (offspring[i] == 0):  # 直到被赋值
                if at:  # 从parent1取基因
                    if temp[parentspring1[at1] - 1] < S:  # parent1对应的这个基因在子代中还没到达最大工序数
                        offspring[i] = parentspring1[at1]  # 赋值
                        offspring[i + W] = parentspring1[at1 + W]
                        offspring[i + W * 2] = parentspring1[at2 + W * 2]
                    at1 += 1  # 不管是否赋值，at1指针向后一格
                    at = not at  # 逻辑取反，下次从parent2取
                else:  # 从parent2取基因
                    if temp[parentspring2[at2]-1] < S:
                        offspring[i] = parentspring2[at2]
                        offspring[i + W] = parentspring2[at2 + W]
                        offspring[i + W * 2] = parentspring2[at2 + W * 2]
                    at2 += 1
                    at = not at  # 逻辑取反
            temp[offspring[i]-1] += 1
        return offspring

    def strategy2_OS(self, s,W):
        parentspring = copy.deepcopy(s)
        length = len(parentspring)//3
        loca1=random.randint(0, length-1)
        loca2=random.randint(0, length-1)
        offspring = copy.deepcopy(parentspring)
        while loca2==loca1:  # 如果相等就重新生成
            loca2 = random.randint(0, length - 1)
        if loca2<loca1:  # 保证loca1<loca2
            temp=loca2
            loca2=loca1
            loca1=temp
        for i in range(loca1,loca2+1):
            offspring[i] = parentspring[loca2 - i + loca1]
            offspring[i + W] = parentspring[loca2 - i + loca1 + W]
            offspring[i + W*2] = parentspring[loca2 - i + loca1 + W*2]
        return offspring

    def strategy3_OS(self, s1, s2,J,W,S):
        parentspring1 = copy.deepcopy(s1)
        parentspring2 = copy.deepcopy(s2)
        choose = [random.choice(range(2)) for i in range(len(s1))]
        temp = np.zeros(J, dtype=int)  # 子代中每个工件已经有几道工序了
        offspring = np.zeros(3 * W, dtype=int)
        offspring[choose] = parentspring1[choose]
        at = 0
        for i in range(len(offspring)//3):
            while (offspring[i] == 0):  # 直到被赋值
                if temp[parentspring2[at] - 1] < S:  # parent1对应的这个基因在子代中还没到达最大工序数
                    offspring[i] = parentspring2[at]  # 赋值
                    offspring[i + W] = parentspring2[at + W]
                    offspring[i + W * 2] = parentspring2[at + W * 2]
                    at += 1  # 不管是否赋值，at1指针向后一格
            temp[offspring[i]-1] += 1
        return offspring

    def strategy1_MS(self,s,s1,s2,W):
        parentspring1 = s1
        parentspring2 = s2
        offspri = copy.deepcopy(s)
        at1 = 0  # parent1指针
        at2 = 0  # parent2指针
        at = True  # 从哪个parent复制
        for i in range(len(s) // 3):
            if at:  # 从parent1取基因
                offspri[i + W] = parentspring1[at1 + W]
                offspri[i + W * 2] = parentspring1[at1 + W * 2]
                at1 += 1  # at1指针向后一格
                at = not at  # 逻辑取反，下次从parent2取
            else:  # 从parent2取基因
                offspri[i + W] = parentspring2[at2 + W]
                offspri[i + W * 2] = parentspring2[at2 + W * 2]
                at2 += 1
                at = not at  # 逻辑取反
        return offspri

    def strategy2_MS(self,s):
        parentspring = s
        offspri = copy.deepcopy(s)
        length = len(parentspring) // 3
        loca1 = random.randint(0, length - 1)
        loca2 = random.randint(0, length - 1)
        while loca2 == loca1:  # 如果相等就重新生成
            loca2 = random.randint(0, length - 1)
        if loca2 < loca1:  # 保证loca1<loca2
            temp = loca2
            loca2 = loca1
            loca1 = temp
        for i in range(loca1, loca2 + 1):
            offspri[i + self.W] = parentspring[loca2 - i + loca1 + self.W]
            offspri[i + self.W * 2] = parentspring[loca2 - i + loca1 + self.W * 2]
        return offspri

    # WS层多点变异
    def mutation(self, parentspring,W,JW):
        offSpring = copy.deepcopy(parentspring)
        length=len(parentspring)//3  #len=36
        num = random.randint(length//6,length//3)  #num个点变异
        for i in range(num): #可能出现随机位置重复
            pos=random.randint(0,length-1)
            mi = parentspring[pos+W]  #对应机器
            wList = JW[int(mi)-1] #对应员工列表
            offSpring[pos+W*2] = np.random.choice(wList)  #随机选择一个员工，可能出现员工没变的情况
        return offSpring

    # 整体移位
    def translocation(self,parentspring,W):
        length = len(parentspring)//3  #self.W
        pos1=random.randint(0,length-1)
        pos2=random.randint(0,length-1)
        offspring=np.zeros(len(parentspring),dtype=int)
        while pos1==pos2:
            pos2 = random.randint(0,length-1)
        if pos1>pos2:  #确保pos1<pos2
            temp=pos1
            pos1=pos2
            pos2=temp
        dist=random.randint(0,pos1)
        for i in range(pos1,pos2):
            offspring[i - dist] = parentspring[i]
            offspring[i + W - dist] = parentspring[i + W]
            offspring[i + W * 2 - dist] = parentspring[i + W*2]

    def generate_next(self,s1,s2,XOVR,MUTR,J,W,S,JW): #两个染色体交叉变异之后选择支配的那个
        if random.random()<XOVR:
            s = self.strategy1_OS(s1, s2,J,W,S)
        else:
            s = s1
        if random.random() < MUTR:
            s = self.strategy1_MS(s, s1, s2,W)
        else:
            s = s
        s = self.mutation(s,W,JW)
        # self.translocation(s)
        return s

    def generate_next1(self,s1, s2,XOVR,MUTR,J,W,S,JW): #两个染色体交叉变异之后选择支配的那个
        if random.random()<XOVR:
            s = self.strategy1_OS(s1, s2,J,W,S)
        else:
            s = self.strategy3_OS(s1, s2,J,W,S)
        if random.random() < MUTR:
            s = self.strategy1_MS(s, s1, s2,W)
        else:
            s = s
        if random.random() < MUTR:
            s = self.mutation(s,W,JW)
        else:
            s = s
        return s

    def cross_mutation1(self,s1,s2,MUTR,XOVR,J,W,S,JmNumber,JW):
        if random.random()<XOVR:
            temp = np.zeros(J, dtype=int)  # 子代中每个工件已经有几道工序了
            offspring = np.zeros(3 * W, dtype=int)
            at1 = 0  # parent1指针
            at2 = 0  # parent2指针
            at = True  # 从哪个parent复制
            for i in range(len(offspring) // 3):
                while (offspring[i] == 0):  # 直到被赋值
                    if at:  # 从parent1取基因
                        if temp[s1[at1] - 1] < S:  # parent1对应的这个基因在子代中还没到达最大工序数
                            offspring[i] = s1[at1]  # 赋值
                            offspring[i + W] = s1[at1 + W]
                            offspring[i + W * 2] = s1[at2 + W * 2]
                        at1 += 1  # 不管是否赋值，at1指针向后一格
                        at = not at  # 逻辑取反，下次从parent2取
                    else:  # 从parent2取基因
                        if temp[s2[at2] - 1] < S:
                            offspring[i] = s2[at2]
                            offspring[i + W] = s2[at2 + W]
                            offspring[i + W * 2] = s2[at2 + W * 2]
                        at2 += 1
                        at = not at  # 逻辑取反
                temp[offspring[i] - 1] += 1
        else:
            offspring = random.choice((s1,s2))
        #变异
        length = W  # len=36
        choice = np.random.choice(length,6,replace=False)
        if random.random()< MUTR:
            for j in range(len(choice)):
                mi = random.randint(1,JmNumber)
                offspring[choice[j] + W] = mi
        choice = np.random.choice(length, 6, replace=False)
        if random.random()< MUTR:
            for j in range(len(choice)):
                mi = offspring[choice[j] + W]
                wList = JW[int(mi) - 1]  # 对应员工列表
                offspring[choice[j] + W * 2] = np.random.choice(wList)  # 随机选择一个员工，可能出现员工没变的情况
        else:
            pass
        return offspring

    def cross_mutation(self,s1,s2,MUTR,XOVR,J,W,S,JmNumber,JW):
        if random.random()<XOVR:
            temp = np.zeros(J, dtype=int)  # 子代中每个工件已经有几道工序了
            offspring = np.zeros(3 * W, dtype=int)
            at1 = 0  # parent1指针
            at2 = 0  # parent2指针
            at = True  # 从哪个parent复制
            for i in range(len(offspring) // 3):
                while (offspring[i] == 0):  # 直到被赋值
                    if at:  # 从parent1取基因
                        if temp[s1[at1] - 1] < S:  # parent1对应的这个基因在子代中还没到达最大工序数
                            offspring[i] = s1[at1]  # 赋值
                            offspring[i + W] = s1[at1 + W]
                            offspring[i + W * 2] = s1[at2 + W * 2]
                        at1 += 1  # 不管是否赋值，at1指针向后一格
                        at = not at  # 逻辑取反，下次从parent2取
                    else:  # 从parent2取基因
                        if temp[s2[at2] - 1] < S:
                            offspring[i] = s2[at2]
                            offspring[i + W] = s2[at2 + W]
                            offspring[i + W * 2] = s2[at2 + W * 2]
                        at2 += 1
                        at = not at  # 逻辑取反
                temp[offspring[i] - 1] += 1
        else:
            offspring = random.choice((s1,s2))
        #变异
        length = W  # len=36
        choice = np.random.choice(length,6,replace=False)
        if random.random()< MUTR:
            for j in range(len(choice)):
                pos = random.randint(0, length - 1)
                mi = random.randint(1,JmNumber)
                offspring[choice[j] + W] = mi
                wList = JW[int(mi) - 1]  # 对应员工列表
                offspring[pos + W * 2] = np.random.choice(wList)  # 随机选择一个员工，可能出现员工没变的情况
        else:
            pass
        return offspring

    def strategy_best(self, s_best, s_best2):
        parentspring1 = copy.deepcopy(s_best)
        parentspring2 = copy.deepcopy(s_best2)
        temp = np.zeros(self.J, dtype=int)  # 子代中每个工件已经有几道工序了
        offspring = np.zeros(3 * self.W, dtype=int)
        x1,x2 = np.random.choice(range(self.W),2,replace=False)
        pos1=min(x1,x2)
        pos2=max(x1,x2)
        for i in range(pos1,pos2): #精英保留
            offspring[i] = parentspring1[i]  # 赋值
            offspring[i + self.W] = parentspring1[i + self.W]
            offspring[i + self.W * 2] = parentspring1[i + self.W * 2]
            temp[offspring[i] - 1] += 1
        at = 0
        # print('1',offspring)
        for j in range(self.W):
            # print('j',j)
            # print('at', at)
            while offspring[j]==0: #我知道了！！可能前面赋值了3个，然后精英保留的时候又有！！
                if temp[parentspring2[at]-1] < self.S:
                    offspring[j] = parentspring2[at]  # 赋值
                    offspring[j + self.W] = parentspring2[at + self.W]
                    offspring[j + self.W * 2] = parentspring2[at + self.W * 2]
                    temp[offspring[j] - 1] += 1
                at += 1

            # print('1',parentspring2[at] - 1)
            # print('2',temp)
            # print('3',offspring[j])
        return offspring


# 7 变邻域搜索
class Search():
    # 倒序变换
    def LS1(self,s,W):
        parentspring = copy.deepcopy(s)
        length = len(parentspring) // 3
        loca1 = random.randint(0, length - 1)
        loca2 = random.randint(0, length - 1)
        offspring = copy.deepcopy(parentspring)
        while loca2 == loca1:  # 如果相等就重新生成
            loca2 = random.randint(0, length - 1)
        if loca2 < loca1:  # 保证loca1<loca2
            temp = loca2
            loca2 = loca1
            loca1 = temp
        for i in range(loca1, loca2 + 1):
            offspring[i] = parentspring[loca2 - i + loca1]
            offspring[i + W] = parentspring[loca2 - i + loca1 + W]
            offspring[i + W * 2] = parentspring[loca2 - i + loca1 + W * 2]
        return offspring

    #任选两个位置交换
    def LS2(self,s,W):
        loca1,loca2 = np.random.choice(len(s)//3,2,replace=False)
        new_s = copy.deepcopy(s)
        temp = np.zeros(3,dtype=int)
        temp[0] = s[loca1]
        temp[1] = s[loca1+W]
        temp[2] = s[loca1+W*2]
        new_s[loca1] = s[loca2]
        new_s[loca1+W] = s[loca2+W]
        new_s[loca1+W*2] = s[loca2+W*2]
        new_s[loca2] = temp[0]
        new_s[loca2 + W] = temp[1]
        new_s[loca2 + W * 2] = temp[2]
        return new_s

    # 任选一个位置变机器
    def LS3(self,s,W,JmNumber):
        loca = np.random.randint(0,len(s) // 3-1)
        new_s = copy.deepcopy(s)
        new_s[loca+W] = random.randint(1, JmNumber)
        return new_s

    # 任选一个位置变人员
    def LS4(self,s,W,JW):
        loca = np.random.randint(0,len(s) // 3-1)
        new_s = copy.deepcopy(s)
        mi = new_s[loca + W]
        wList = JW[int(mi) - 1]  # 对应员工列表
        new_s[loca + W*2] = np.random.choice(wList)
        return new_s

    # 在最后一道工序替换最小工作负载机器
    def LS5(self,s,J,W,TMF):
        Sel = Select()
        new_s = copy.deepcopy(s)
        job = random.randint(0,J-1)
        pos = 0
        for pos_i in range(len(s)//3-1,-1,-1):
            if s[pos_i] == job:
                pos = pos_i
                break
        # new_s[pos+W] = np.argmin(TMF) + 1
        new_s[pos + W] = Sel.roulette(TMF) + 1
        return new_s

    # 在最后一道工序替换最小疲劳度人员
    def LS6(self, s, J, W, fatigue):
        Sel = Select()
        new_s = copy.deepcopy(s)
        job = random.randint(0, J - 1)
        pos = 0
        for pos_i in range(len(s) // 3 - 1, -1, -1):
            if s[pos_i] == job:
                pos = pos_i
                break
        # new_s[pos + W*2] = np.argmin(fatigue) + 1
        new_s[pos + W * 2] = Sel.roulette(fatigue) + 1
        return new_s

# 8 强化学习
# class Q_learn(temp):



