import pandas as pd
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
# from sklearn import preprocessing
from sklearn import svm
import urllib
import pickle
from scipy.sparse import csr_matrix

def stemme(text):
    s = set()
    st = LancasterStemmer()
    for word in text:
        sword = st.stem(word)
        s.add(sword)
    s= sorted(s)
    return s

def lemmatizeme(text):
    s = set()
    d ={}
    lmtzr = WordNetLemmatizer()
    for word in text:
        lword = lmtzr.lemmatize(word)
        print word+" : "+lword
        s.add(lword)
        # d[word]=lword
    s= sorted(s)
    return s

def labelExtraction(filename='../Data/consolidatedlabels.csv'):
    df = pd.read_csv(filename,name=['labels'])
    return df['labels']

def featureExtraction(filename='../Data/consolidatednews.csv',dicfilename='../Data/dictionary.csv'):
    df = pd.read_csv(filename,names = ['date','URL'])
    df2 = pd.read_csv(dicfilename,names = ['Words'])

    dic = df2['Words']
    # dic =lemmatizeme(dic)
    dic = stemme(dic)
    u = df['URL'].values
    cv = CountVectorizer(vocabulary=dic)
    alltext=[]
    i = 1
    print "Total :{}".format(len(u))
    for link in u:
    	print i
    	i+=1
        f =urllib.urlopen(link)
        text = f.read()
        text = unicode(text, "utf-8")
        text =stemme(text)
        text = ''.join(text)
        alltext.append(text)
    xtraincv = cv.fit_transform(alltext)
    X=xtraincv.toarray()
    return X

def labelExtraction(filename="../Data/consolidatedlabels.csv"):
    df = pd.read_csv(filename, names=['Labels'])
    y = df['Labels']
    return y

def LR(X,y,split = 0.1):

    scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
    print "LR ->"
    print scores
    print scores.mean()
    return scores.mean()


def SVM(X,y,split =0.2):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    model = svm.SVC(kernel='linear')
    # model.fit(X, y)
    # model.score(X, y)
    # # Predict Output
    # predicted = model.predict(x_test)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=10)
    print "SVM->"
    print scores
    print scores.mean()
    return scores.mean()

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)



def NN(X,y,split =0.2):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    # # Variable initialization
    # epoch = 5000  # Setting training iterations
    # lr = 0.1  # Setting learning rate
    # X= X_train
    # inputlayer_neurons = X.shape[1]  # number of features in data set
    # hiddenlayer_neurons = 3  # number of hidden layers neurons
    # output_neurons = 1  # number of neurons at output layer
    # # weight and bias initialization
    # wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
    # bh = np.random.uniform(size=(1, hiddenlayer_neurons))
    # wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
    # bout = np.random.uniform(size=(1, output_neurons))
    # for i in range(epoch):
    #
    #     # Forward Propogation
    #     hidden_layer_input1=np.dot(X,wh)
    #     hidden_layer_input=hidden_layer_input1 + bh
    #     hiddenlayer_activations = sigmoid(hidden_layer_input)
    #     output_layer_input1=np.dot(hiddenlayer_activations,wout)
    #     output_layer_input= output_layer_input1+ bout
    #     output = sigmoid(output_layer_input)
    #
    #     #Backpropagation
    #     E = y_train-output
    #     slope_output_layer = derivatives_sigmoid(output)
    #     slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    #     d_output = E * slope_output_layer
    #     Error_at_hidden_layer = d_output.dot(wout.T)
    #     d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    #     wout += hiddenlayer_activations.T.dot(d_output) *lr
    #     bout += np.sum(d_output, axis=0,keepdims=True) *lr
    #     wh += X.T.dot(d_hiddenlayer) *lr
    #     bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
    #
    # print output
    # return output
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
    print "NN->"
    print scores
    print scores.mean()
    return scores.mean()

def KNN(X,y,split =0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    # # instantiate learning model (k = 3)
    # knn = KNeighborsClassifier(n_neighbors=3)
    # # fitting the model
    # knn.fit(X_train, y_train)
    # # predict the response
    # pred = knn.predict(X_test)
    #
    # # evaluate accuracy
    # print accuracy_score(y_test, pred)
    myList = list(range(1, 50))
    # subsetting just the odd ones
    neighbors = filter(lambda x: x % 2 != 0, myList)
    # empty list that will hold cv scores
    cv_scores = []
    # perform 10-fold cross validation
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    # print max(cv_scores)
    print "KNN ->"
    print max(cv_scores)
    return max(cv_scores)

if __name__ == "__main__":
    # X = featureExtraction()
    # import pickle
    # fout = open("features.pickle","w")
    # X = np.array(X)
    # pickle.dump(X,fout)
    # fout.close()
    X = None
    for name in [ "jigar","yang","soundarya","manoj","ashish"]:
        X1 = pickle.load(open("../Data/NewsPickle" + "featrues_"+name+"_3000.pickle"))
        X2 = pickle.load(open("../Data/NewsPickle" + "featrues_"+name+"_6000.pickle"))
        X3 = pickle.load(open("../Data/NewsPickle" + "featrues_"+name+"_9000.pickle"))
        X1 = np.array(X1.todense())
        X2 = np.array(X2.todense())
        X3 = np.array(X3.todense())
        if X is None:
            X = np.concatenate((X1,X2,X3),axis = 0)
        else:
            X = np.concatenate((X,X1,X2,X3),axis = 0)
    meanX = np.mean(X, axis=1)
    SDX = np.std(X,axis=1)
    n,m = X.shape
    for i in range(n):
        X[i][:] =(X[i][:] - meanX[i])/SDX[i]
    # X = (X-meanX)/SDX
    # X_normalized = preprocessing.normalize(X, norm='l2')"
    y=labelExtraction()
    acc_Logistic = LR(X,y)
    acc_SVM = SVM(X,y)
    avg_NN  = NN(X,y)
    avg_KNN = KNN(X,y)

    # print " LR :- " + acc_Logistic + " SVM :- " + acc_SVM + " NN :-" + avg_NN + " KNN :- " + avg_KNN




