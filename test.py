import random
import numpy as np
import time
from multiprocessing import Pool
from threading import Timer
import multiprocessing
from scipy import integrate
import pandas as pd
import openpyxl
import xlrd
# from algori_func import *
import matplotlib.pyplot as plt


# JM = np.array([[[1,2,3],[1,2,3],[9],[9],[1,2,3],[1,2,3],[1,2,3]],
#                    [[8],[1,2,3],[1,2,3],[1,2,3],[8],[9],[4,5]],
#                   [[4,5],[8],[6,7],[8],[1,2,3],[6,7],[9]],
#                    [[9],[9],[8],[1,2,3],[6,7],[8],[9]],
#                    [[6,7],[1,2,3],[9],[6,7],[9],[1,2,3],[1,2,3]],
#                     [[9],[6,7],[4,5],[1,2,3],[6,7],[6,7],[8]],
#                      [[9],[6,7],[6,7],[6,7],[4,5],[4,5],[6,7]]])
# JM = np.array([[[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
#                    [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
#                   [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
#                    [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
#                    ])

# f_1 = 1
# a = 0.5
# def f_x(i):
#     f_x_i = 0.5 ** i * f_1 + 2 * (1 - a ** i) / (1 - a)
#     return f_x_i
# v, err = integrate.quad(f_x, 1, 2)
# print(v)
# start = time.time()
# with Pool() as p:
#     # with Pool(4) as p: # 指定4个进程
#     data_infos = p.map(f_x, range(10000000))
# del data_infos
# end = time.time()
# print(end-start)
# if __name__ == '__main__':
#     a = 0.5
#     f_1 = 1
#     f = 1
#     f_x = np.zeros(10000000)
#
#     # start = time.time()
#     # for i in range(10000000):
#     #     f = a ** i * f + 2 * (1 - a ** i) / (1 - a)
#     # end = time.time()
#     # print(end - start)
#     #
#     # start = time.time()
#     # for i in range(10000000):
#     #     f_x[i] = a ** i * f_1 + 2 * (1 - a ** i) / (1 - a)
#     # end = time.time()
#     # print(end - start)
#     start = time.time()
#     v, err = integrate.quad(f_x, 1, 2)
#     end = time.time()
#     print(v)
#     print(end - start)
#     f_x = list(f_x)
#     start = time.time()
#     pool= multiprocessing.Pool(processes=4)
#     r = pool.map(f_x, range(10000000))
#     pool.close()
#     end = time.time()
#     print(r)
#     print(end-start)
#
#
# # P = decode(OS)
# # for k in range(parse.W):
# #     val = int(P[k])
# #     a = int(val % 100)  # 工序
# #     b = int((val - a) / 100)  # 工件
# #     tmp_m = parse.JM[a][b]  ##可选机器
# #     MS[k] = np.random.choice(tmp_m)  ##随机选择一个机器
# #     tmp_w = parse.JW[MS[k] - 1]  ##机器对应的可选人员
# #     WS[k] = np.random.choice(tmp_w)  ##随机选择一个人员
#
#
#



# open('result.xlsx', encoding='gbk')
# c= pd.read_excel(io='./result.xlsx')
# c = np.array(c)
# c = c[5:]
# imoead = c[:,0]
# moead = c[:,1]
# nsga2 = c[:,2]
# spea2 = c[:,3]
# for i in range(len(imoead)):
#     imoead[i] = np.array(imoead[i])
#     moead[i] = np.array(moead[i])
#     nsga2[i] = np.array(nsga2[i])
#     spea2[i] = np.array(spea2[i])
# print(imoead[0])



# 读三列
def Read_cell(sheet_num):
    # 导入需要读取Excel表格的路径
    data = xlrd.open_workbook('result.xls')
    table = data.sheets()[sheet_num]
    # 创建一个空列表，存储Excel的数据
    tables = np.zeros((60,3))
    # 将excel表格内容导入到tables列表中def
    # for rown in range(table.nrows):
    for rown in range(6,66):
        tables[rown - 7, 0] = table.cell_value(rown, 64)
        tables[rown - 7, 1] = table.cell_value(rown, 65)
        tables[rown - 7, 2] = table.cell_value(rown, 66)
    return tables

# 读一行
def Read_line(sheet_num,obj): #obj改变目标值，1，2，3
    # 导入需要读取Excel表格的路径
    # data = xlrd.open_workbook('result.xls')
    # table = data.sheets()[0]
    data = openpyxl.load_workbook('result_.xlsx')
    sheet = data.worksheets[sheet_num]
    # table = data.sheets()[sheet_num]
    # 创建一个空列表，存储Excel的数据
    tables = []
    # 将excel表格内容导入到tables列表中def
    for i in range(41):
        tables.append((sheet.cell(row=72, column=obj+i*4)).value)
        # tables.append(table.cell_value(69, obj+i*4))
    # tables.append(table.cell_value(rown, 1))
    return tables


# x = list(range(0,205,5))
# print(x)
# # y1 = [1247,931,883,877,795,768,752,791,782,816]
# # y2 = [1100,990,880,890,770,889,770,776,880,890]
# y1 = Read_line(0,1)
# y2 = Read_line(1,1)
# y3 = Read_line(2,1)
# y4 = Read_line(3,1)
#
# plt.title('Convergence curve')  # 折线图标题
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
# plt.xlabel('Iterations')  # x轴标题
# plt.ylabel('Makespan')  # y轴标题
# plt.plot(x, y1, 'b*--', alpha=0.5, linewidth=1, label='IMOEAD')  # 'bo-'表示蓝色实线，数据点实心原点标注
# plt.plot(x, y2, 'g^--', alpha=0.5, linewidth=1, label='MOEAD')
# plt.plot(x, y3, 'rs--', alpha=0.5, linewidth=1, label='NSGA2')
# plt.plot(x, y4, 'yo--', alpha=0.5, linewidth=1, label='SPEA2')
# plt.legend()  #显示上面的label
# plt.show()


# y1 = Read_cell(0)
# y2 = Read_cell(1)
# y3 = Read_cell(2)
# y4 = Read_cell(3)
# print(Coverage(y1,y2))
# print(Coverage(y2,y1))
# print(Coverage(y1,y3))
# print(Coverage(y3,y1))
# print(Coverage(y1,y4))
# print(Coverage(y4,y1))



# 创建数据
x = np.arange(3)

# 有a/b两种类型的数据，n设置为2
total_width, n = 0.6, 3
# 每种类型的柱状图宽度
width = total_width / n
# open('result.xls', encoding='gbk')
# data = pd.read_excel(io='./result.xls',sheet_name='NONE')
# data = np.array(data)
# data = data[6:65] #下面数据部分
moead = [974,1337.19,230.6]
metropolis = [745,1316.86,216.1729028]
operator = [680,1254.06,77.49481689]
tsea = [632,1229.7,76.33024206]

# 重新设置x轴的坐标
x = x - (total_width - width) / 2
# print(x)
plt.rcParams['font.serif'] = ['Times New Roman']
# 画柱状图
plt.bar(x, moead, width=width, label="MOEA/D", color='salmon')
plt.bar(x + width, metropolis, width=width, label="TOP-Metro", color='darkseagreen')
plt.bar(x + 2*width, operator, width=width, label="Perturbation ",color='navajowhite')
plt.bar(x + 3*width, tsea, width=width, label="TSEA",color='thistle')
plt.xticks(np.arange(3), ('Cmax', 'Pc', 'Lb'))
# 显示图例
# plt.figure(dpi=300,figsize=(24,24))
plt.legend(loc='upper right', prop={"family": "Times New Roman"})
plt.xlabel("Comparison of Algorithm Improvements", fontname="Times New Roman")
plt.ylabel("Objective Values", fontname="Times New Roman")
# plt.savefig('plot123_2.png', dpi=500)
# 显示柱状图
plt.show()


