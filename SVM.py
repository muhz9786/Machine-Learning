from sklearn.svm import SVC as SVM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
import pandas as pd
import time

# 读取数据
f = pd.read_csv("./data/amazon_cells.csv")
x = f.iloc[:, 0]
y = f.iloc[:, 1]
print('未处理的第一个样本的属性为：')
print(x[0])

# 构建词袋
vect = CountVectorizer()
x = vect.fit_transform(x)
print('词袋处理后的第一个样本的属性为：')
print(x[0])

# 构建TF-IDF特征，归一化
tf_transformer = TfidfTransformer().fit(x)
x = tf_transformer.transform(x)
print('再经TF-IDF处理后的第一个样本的属性为：')
print(x[0])

# 随机分类训练集与测试集
trainX, testX, trainY, testY = train_test_split(
    x, y, test_size=0.2, random_state=1)

# 支持向量机
modle = SVM(C=1000.0, kernel='rbf', random_state=1)
start = time.perf_counter()
modle.fit(trainX, trainY)
end = time.perf_counter()
print('Running time: %.12s ms' % ((end - start) * 1000))

# 输出评估结果
expected = testY
predict = modle.predict(testX)
print('Report:')
print(metrics.classification_report(expected, predict))

# 建立混淆矩阵
lable = list(set(expected))
matrix = pd.DataFrame(metrics.confusion_matrix(
    expected, predict, labels=lable), index=lable, columns=lable)
print('Matrix:')
print(matrix)
