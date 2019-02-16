from sklearn.neural_network import MLPClassifier as MLP
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import pickle
import time
from PIL import Image

# 读取数据
f = open('./data/mnist.pkl', 'rb')
train, valid, test = pickle.load(f, encoding='bytes')

# 划分数据
trainX = train[0]
trainY = train[1]
testX = test[0]
testY = test[1]

# 显示第一个手写数字的图片
I = trainX[0]
I.resize((28, 28))
im = Image.fromarray((I*256).astype('uint8'))
print('第一个实例为：%s ,图片已生成。' % trainY[0])
im.show()

# 神经网络
start = time.time()
modle = MLP(activation='logistic', solver='sgd',
            learning_rate_init=0.5, max_iter=100)
modle.fit(trainX, trainY)
end = time.time()
print('Running time: %s s' % (end-start))

# 输出评估结果
expected = testY
predict = modle.predict(testX)
print(metrics.classification_report(expected, predict))

# 建立混淆矩阵
lable = list(set(expected))
matrix = pd.DataFrame(metrics.confusion_matrix(
    expected, predict, labels=lable), index=lable, columns=lable)
print(matrix)
