# 读入数据
import pandas as pd
# 划分训练集和测试集，使得测试集中包含1000条数据
from sklearn import model_selection
url = 'class-info-test.csv'
wine = pd.read_csv(url)

# 生成X和y
feature_cols = [ '人数', '学时', '楼层', '周一', '周二', '周三', '周五']
x = wine[feature_cols]
y = wine.target
# 检查特征的数据类型
print(wine.shape)
print(wine.info())
print(wine.target)
wine.head()

df = pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1) #将特征和标签合并，axis=0为横向合并
print(df)
exit()
x_train , x_test, y_train, y_test = model_selection.train_test_split(x, y , test_size=100, random_state=1)
print(x_test.shape)
x_test.head()
