import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn import cross_validation
import sklearn 

csv_data = pd.read_csv('C:/Users/administrator.FANEDATA/Desktop/Tensorflow/999_python_classification/data_ana.csv')
csv_data.dropna(axis=0, how='any', inplace=True)
print(csv_data.shape)
csv_data.tail(5)
csv_data.head(5)

mmm,csv_data=np.split(csv_data,indices_or_sections=(1,),axis=1)
x,y=np.split(csv_data,indices_or_sections=(5,),axis=1) #x为数据，y为标签
x = x.as_matrix()
y = y.as_matrix()
m,n=y.shape
y=y.reshape(m)

n_samples, n_features = x.shape

random_state = np.random.RandomState(0)
x = np.c_[x, random_state.randn(n_samples, 200 * n_features)]

train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x,y, random_state=20190705, train_size=0.7,test_size=0.3)

svm = svm.SVC(kernel='linear', probability=True,random_state=random_state)

y_score = svm.fit(train_data,train_label).decision_function(test_data)


fpr,tpr,threshold = roc_curve(test_label, y_score)
roc_auc = auc(fpr,tpr)

plt.figure() 
lw = 2 
plt.figure(figsize=(10,10)) 
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') 
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05]) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('Receiver operating characteristic example') 
plt.legend(loc="lower right") 
plt.show()
