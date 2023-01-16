# -*- coding: utf-8 -*-
# 读入数据
import pandas as pd
import pydotplus as pydotplus


filepath = 'class-info-dataset.csv'
wine = pd.read_csv(filepath)

# 生成x和y
feature_names = ["周编号", "周天", "人数", "学时", "是否空闲", "楼层"]
target_names = "target_name"
features_data = wine[feature_names]
class_names = wine[target_names]

targets = pd.DataFrame(class_names)
class_names_vals = []
for i in class_names.to_dict().values():
    class_names_vals.append(i)

# # （4）划分训练集和测试集
from sklearn.model_selection import train_test_split
import numpy as np

x_data = np.array(features_data)  # 类别数据
y_data = np.array(targets)  # 特征数据

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10)

# （5）使用决策树方法进行分类
from sklearn.tree import DecisionTreeClassifier

# 接收决策树分类器
clf = DecisionTreeClassifier()
# regr_1 = DecisionTreeClassifier(max_depth=2)
# regr_2 = DecisionTreeClassifier(max_depth=5)
# 训练数据输入,fit()只能识别数值类型，或sparse矩阵
clf.fit(x_train, y_train)
# regr_1.fit(x_train, y_train)
# regr_2.fit(x_train, y_train)
# 评分法，准确率
accuracy = clf.score(x_test, y_test)

print('测试集打分', accuracy)
print('训练集打分', clf.score(x_train, y_train))
# 预测

print("xtest",x_test,type(x_test))
# _test = [[10,4,40,126,1,4]]
y_pred = clf.predict(x_test)

# y_1 = regr_1.predict(x_test)
# y_2 = regr_2.predict(x_test)

# probability = clf.predict_proba(x_test)
print("预测值：", y_pred)

# 可视化
# 在训练集上训练一个没有对树的最大深度、特征及叶子节点等方面有任何限制的决策树分类器
# 绘制并显示决策树
import graphviz
from sklearn import tree

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_names,
                                class_names=class_names_vals,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("dtree.pdf")

