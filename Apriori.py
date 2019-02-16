import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
import time

# 读取数据
store_data = pd.read_csv('./data/store_data.csv', header=None)

# 处理数据
records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i, j]) for j in range(0, 20)])

# 关联规则
start = time.perf_counter()
association_rules = apriori(
    records, min_support=0.005, min_confidence=0.3, min_lift=3, min_length=2)
end = time.perf_counter()
print('Running time: %s ms' % ((end - start) * 1000))
association_results = list(association_rules)

# 输出
print('发现 {} 条规则'.format(len(association_results)))
for item in association_results:
    pair = item[0]
    items = [x for x in pair]
    print("规则: " + items[0] + " -> " + items[1])
    print("支持度: " + str(item[1]))
    print("置信度: " + str(item[2][0][2]))
    print("提升度: " + str(item[2][0][3]))
    print("=====================================")
