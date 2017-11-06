import pandas as pd

df_train = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-train.csv')   #训练文件
df_test = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-test.csv')     #测试文件

#测试集正负样本分类，特征选取
df_test_negative = df_test.loc[df_test['Type']==0][['Clump Thickness','Cell Size']]
df_test_positive = df_test.loc[df_test['Type']==1][['Clump Thickness','Cell Size']]
import matplotlib.pyplot as plt

#绘制点
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],s=200,c='red',marker='o')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')

#绘制x,y
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')

#显示
#plt.show()

import numpy as np

intercept=np.random.random([1])
coef=np.random.random([2])

lx=np.arange(0,12)
ly=(-intercept-lx*coef[0])/coef[1]

plt.plot(lx,ly,c='yellow')

plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],s=200,c='red',marker='o')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')

plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')

#plt.show()

#sklearn逻辑回归分类器
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

#训练
lr.fit(df_train[['Clump Thickness','Cell Size']][:10],df_train['Type'][:10])
print('Testing accuracy(10 training samples):',lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))

intercept = lr.intercept_
coef = lr.coef_[0,:]

ly = (-intercept-lx*coef[0])/coef[1]

plt.plot(lx,ly,c='green')

plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],s=200,c='red',marker='o')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')

plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')

#plt.show()

lr = LogisticRegression()

#训练
lr.fit(df_train[['Clump Thickness','Cell Size']],df_train['Type'])
print('Testing accuracy(10 training samples):',lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))

intercept = lr.intercept_
coef = lr.coef_[0,:]

ly = (-intercept-lx*coef[0])/coef[1]

plt.plot(lx,ly,c='blue')

plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],s=200,c='red',marker='o')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')

plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')

plt.show()