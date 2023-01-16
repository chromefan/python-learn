#-*- coding: utf-8 -*-
import pandas as pd
import pydotplus

#参数初始化

dt = pd.read_excel('good-gua.xlsx',sheet_name='挑瓜') #导入数据

#去掉编号列，同时将好瓜这一列名改为是否，这样生成的图里更容易看明白
dt.rename(columns={'好瓜':'是否'},inplace=True)
dt1=dt.iloc[:,1:]
print(dt1)
#定义一个数据转换的函数，将中文数据转换为数值
def convert(dt):
    if dt in ['青绿','蜷缩','浊响','清晰','凹陷','硬滑','是']:
        return 1
    elif dt in ['乌黑','稍蜷', '沉闷', '稍糊', '稍凹', '软粘','否']:
        return 2
    elif dt in ['浅白', '硬挺', '清脆', '模糊', '平坦']:
        return 3
dt1=dt1.applymap(convert)

import numpy as np
labels=np.array(dt1['是否'])#特征数据
data=np.array(dt1.iloc[:,:-1])#类别数据

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion = 'entropy')#参数criterion = 'entropy'为基于信息熵，‘gini’为基于基尼指数
clf.fit(data, labels)#训练模型


print(data)
exit()
dot_data = tree.export_graphviz(clf, out_file=None,
                     feature_names=dt1.columns[:-1].values,
                     class_names=dt1.columns[-1],
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("gua.png")

import graphviz
t=['乌黑','蜷缩','清脆','模糊','平坦','硬滑']
t=pd.DataFrame(t).T.applymap(convert)
r = clf.predict(t)
print(r)