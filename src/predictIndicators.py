import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split



data = None
data2 = None
labels = None

for f in os.listdir(os.curdir + "./Data/Indicators/"):
    temp = np.genfromtxt(f,delimiter = ",")
    lable = temp[:,-1]
    temp1 = temp[:,:-1]
    temp2 = temp[:,:-2]
    if data is None:
        data =temp1
        data2 = temp2
        labels = label
    else:
        data = np.vstack([data,temp1])
        data2 = np.vstack([data2,temp2])
        labels = np.hstack([labels,label])

X_train,X_test,y_train,y_test = train_test_split(data,labels,train_size = 0.8)
clf = MLPClassifier(hidden_layers_sizes = (5,))
clf.fit(X_train,y_train)
print "Score with News : {}".format(clf.score(X_test,y_test))


X_train,X_test,y_train,y_test = train_test_split(data2,labels,train_size = 0.8)
clf2 = MLPClassifier(hidden_layers_sizes= (5,))
clf2.fit(X_train,y_train)
print "Score without news : {}".format(clf.score(X_test,y_test))