# _*_ encoding:utf-8 _*_
import numpy as np
from collections import Counter

# 计算决策树中 entropy 系数
def entropy(x):
    """
    :param x: 各分类数量 
    :return: entropy
    """
    x = np.array(x)
    value = 0
    for i in x:
        if i == 0:
            continue
        else:
            value += -i/x.sum()*np.log2(i/x.sum())

    return value

def info(c, x, y, type):
    """
    :param c: constant 父节点熵
    :param x: True: 各分类数量
    :param y: False: 各分类数量
    :return:  δinfo
    """
    x = np.array(x)
    y = np.array(y)
    N = x.sum()+y.sum()
    if type == 'entropy':
        return c-x.sum()/N*entropy(x)-y.sum()/N*entropy(y)
    elif type == 'gini':
        return c-x.sum()/N*gini(x)-y.sum()/N*gini(y)

# 计算决策树中 gini 系数
def gini(x):
    """
    :param x:  True：各分类数量
    :return: gini
    """
    x = np.array(x)
    return 1-(x[0]/x.sum())**2-(x[1]/x.sum())**2

# 信息增益--熵
# constant = entropy([4,6])
# A = info(constant, [4,3], [0,3], 'entropy')
# B = info(constant, [3,1], [1,5], 'entropy')

# 信息增益--基尼
# constant = gini([4,6])
# A = info(constant, [4,3], [0,3], 'gini')
# B = info(constant, [3,1], [1,5], 'gini')

n = np.array([898, 690, 205, 699, 303, 690,
              768, 1000, 214, 270, 155, 368,
              351, 150, 57, 3200, 148, 768,
              208, 958, 846, 178, 101])
dt = np.array([.9209, .8551, .8195, .9514, .7624, .8580,
               .7240, .709, .6729, .80, .8194, .8533,
               .8917, .9467, .7895, .7334, .7703, .7435,
               .7885, .8372, .7104, .9438, .9307])
by = np.array([.7962, .7681, .5805, .9599, .8350, .7754,
               .7591, .7470, .4859, .8407, .8323, .7880,
               .8234, .9533, .9474, .7316, .8311, .7604,
               .6971, .7004, .4504, .9663, .9307])
vt = np.array([.8719, .8478, .7073, .9642, .8449, .8507,
               .7682, .7440, .5981, .8370, .8710, .8261,
               .8889, .96, .9298, .7356, .8649, .7695,
               .7692, .9833, .7494, .9888, .9604])

def classifier_compare(pa, pb, n):
    p = (pa+pb)/2
    z = (pa-pb)/(2*p*(1-p)/n)**0.5
    res = [1 if i>1.96 else -1 if i<-1.96 else 0 for i in z]
    return Counter(res),z

db = classifier_compare(dt, by, n)
dv = classifier_compare(dt, vt, n)
bv = classifier_compare(by, vt, n)