import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn import cross_validation
import sklearn 

from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus


csv_data = pd.read_csv('C:/Users/administrator.FANEDATA/Desktop/Tensorflow/999_python_classification/data_ana.csv')
csv_data.dropna(axis=0, how='any', inplace=True)
mmm,csv_data=np.split(csv_data,indices_or_sections=(1,),axis=1) 
x,y=np.split(csv_data,indices_or_sections=(5,),axis=1) #x为数据，y为标签

#------------------------------------------------

n_samples, n_features = x.shape

random_state = np.random.RandomState(0)
x = np.c_[x, random_state.randn(n_samples, 200 * n_features)]

#------------------------------------------------


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=20190708)


clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
y_score=clf.fit(x_train, y_train).predict(x_test)


fpr,tpr,threshold = roc_curve(y_test, y_score)
roc_auc = auc(fpr,tpr)


plt.figure() 
lw = 2 
plt.figure(figsize=(10,10)) 
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) #
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') 
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05]) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('Receiver operating characteristic example') 
plt.legend(loc="lower right") 
plt.show()
