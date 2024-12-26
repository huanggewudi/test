import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'booster': 'gbtree',  # 使用树模型
    'objective': 'multi:softprob',  # 多分类问题
    'num_class': 3,  # 类别数
    'eta': 0.3,  # 学习率
    'max_depth': 6,  # 树的最大深度
    'eval_metric': 'mlogloss'  # 评价指标
}

num_round = 100  # 迭代次数

bst = xgb.train(params, dtrain, num_round)

preds = bst.predict(dtest)

# 获取预测的类别
best_preds = [int(np.argmax(line)) for line in preds]

accuracy = accuracy_score(y_test, best_preds)
print(f"Accuracy: {accuracy * 100:.2f}%")

importance = bst.get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': importance.keys(),
    'Importance': importance.values()
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# cv_results = xgb.cv(
#     params,
#     dtrain,
#     num_boost_round=num_round,
#     nfold=5,  # 5折交叉验证
#     metrics=['mlogloss'],
#     early_stopping_rounds=10
# )
# print(cv_results)
