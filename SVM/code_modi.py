from sklearn import svm
import pandas as pd
import numpy as np
import sklearn 

csv_data = pd.read_csv('C:/Users/administrator.FANEDATA/Desktop/Tensorflow/999_python_classification/data_ana.csv')
csv_data.dropna(axis=0, how='any', inplace=True)
print(csv_data.shape)
csv_data.tail(5)
csv_data.head(5)

def flag_label(s):
    it={b'non_PBC':0, b'PBC':1}
    return it[s]
	
mmm,csv_data=np.split(csv_data,indices_or_sections=(1,),axis=1) 
x,y=np.split(csv_data,indices_or_sections=(5,),axis=1) #x为数据，y为标签
x = x.as_matrix()


y = y.as_matrix()
m,n=y.shape
y=y.reshape(m)


train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x,y, random_state=20190705, train_size=0.7,test_size=0.3)


classifier=svm.SVC(C=2,kernel='rbf',gamma=10,decision_function_shape='ovr') 
classifier.fit(train_data,train_label.ravel())


print("训练集：",classifier.score(train_data,train_label))
print("测试集：",classifier.score(test_data,test_label))

from sklearn.metrics import accuracy_score
tra_label=classifier.predict(train_data) #训练集的预测标签
tes_label=classifier.predict(test_data) #测试集的预测标签
print("训练集：", accuracy_score(train_label,tra_label) )
print("测试集：", accuracy_score(test_label,tes_label) )
