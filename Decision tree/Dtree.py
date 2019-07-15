import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus
import pandas as pd

# 1、read and split data
csv_data = pd.read_csv('C:/Users/administrator.FANEDATA/Desktop/Tensorflow/999_python_classification/data_ana.csv')
csv_data.dropna(axis=0, how='any', inplace=True)
mmm,csv_data=np.split(csv_data,indices_or_sections=(1,),axis=1) 
x,y=np.split(csv_data,indices_or_sections=(5,),axis=1) #x:data，y:labels

# 2、split data (both data and labels) into training and testing parts
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=20190708)

# 3、training 
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(x_train, y_train)


play_feature_E = 'idn', 'L31', 'L55', 'L155', 'gender', 'AGE'
play_class = 'yes', 'no'

# 4、structure of tree
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=play_feature_E, class_names=play_class,filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('C:/Users/administrator.FANEDATA/Desktop/Tensorflow/999_python_classification/DT/code/play1.pdf')

print(clf.feature_importances_)

# 5、prediction (optional)
answer = clf.predict(x_train)
y_train = y_train.as_matrix()
y_train = y_train.reshape(-1)
print(answer)
print(y_train)
print(np.mean(answer == y_train))

# 6、prediction of test data
answer = clf.predict(x_test)
y_test = y_test.as_matrix()
y_test = y_test.reshape(-1)
print(answer)
print(y_test)
print(np.mean(answer == y_test))
