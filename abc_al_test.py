import matplotlib.pyplot as plt
import numpy as np
from ypstruct import structure
import time
import artificial_bee_colony

start = time.time()         #运行开始时刻
# 测试函数
def sphere(x):
    return sum(x**2)

# 问题定义
problem = structure()
problem.costfunc = sphere
problem.nvar = 3
problem.varmin = -100 * np.ones(3)
problem.varmax = 100 * np.ones(3)

# ABC参数
params = structure()
params.maxit = 40
params.npop = 50
params.nonlooker = 100
params.a = 1

# 运行ABC
out = artificial_bee_colony.run(problem, params)
# 运行结果
plt.rcParams['font.sans-serif'] = ['KaiTi']  #设置字体为楷体
plt.plot(out.bestcost)
print("最优解：{}".format(out.bestsol))
end = time.time()              # 运行结束时刻
print('运行时间：{}s'.format(end-start))

plt.xlim(0, params.maxit)
plt.xlabel('迭代次数')
plt.ylabel('全局最优目标函数值')
plt.title('人工蜂群算法')
plt.grid(True)
plt.show()