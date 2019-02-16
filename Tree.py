from sklearn.tree import DecisionTreeClassifier as TREE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import time

# 读取数据
f = pd.read_csv("./data/Watermelon2.0.csv", encoding='gbk')
print(f)
x = f.iloc[:, 0:6]
y = f.iloc[:, 6]

# 处理标签属性
encoder = LabelEncoder()
for i in ('色泽', '根蒂', '敲声', '纹理', '脐部', '触感'):
    x[i] = encoder.fit_transform(x[i])
print('处理后的属性值:\n{}'.format(x))

# 归一化（使用极差法）
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
print('归一化后的属性值：\n{}'.format(x))

# 随机分类训练集与测试集
trainX, testX, trainY, testY = train_test_split(
    x, y, test_size=0.2, random_state=1)

# 决策树模型
start = time.perf_counter()
modle = TREE(criterion='entropy')
modle.fit(trainX, trainY)
end = time.perf_counter()
print('Running time: %s ms' % ((end-start) * 1000))

# 输出评估结果
expected = testY
predict = modle.predict(testX)
print(metrics.classification_report(expected, predict))

# 建立混淆矩阵
lable = list(set(expected))
matrix = pd.DataFrame(metrics.confusion_matrix(
    expected, predict, labels=lable), index=lable, columns=lable)
print(matrix)
