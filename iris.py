



import pandas  as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.ensemble import AdaBoostClassifier
from sklearn import model_selection
#from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
vatchala={
        'Support':SVC(),
          'Rand':RandomForestClassifier(),
          'Tree':DecisionTreeClassifier(),
          'Naive':GaussianNB()
         }






#names=["sepal_length" , "sepal_width","petal_length","petal_width","class"]
#dataset=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",names=names)
dataset=pd.read_csv("D.csv")
dataset.shape
dataset.tail
dataset.head
dataset.describe
dataset.describe()
input_columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
#input_columns=[:,0:4]
output_columns=["Species"]
#output_columns=[:,4]
#X=dataset.iloc[: ,:-1].values
#print (X)
#Y=dataset.iloc[:,-1].values
#print (Y)
X=dataset[input_columns].values
Y=dataset[output_columns].values
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.30,random_state=5)
vatchala={
        'Support':SVC(),
          'Rand':RandomForestClassifier(),
          'Tree':DecisionTreeClassifier(),
          #'Naive':GaussianNB(),
        'AdaBoost':AdaBoostClassifier()
         }
for N,V in vatchala.items():
    nb_model=V
    nb_model.fit(X_train,Y_train)
    nb_predict_train=nb_model.predict(X_train)
    nb_predict_test=nb_model.predict(X_test)
    print(N,"Accuracy:{0:.4f}".format(metrics.accuracy_score(nb_predict_test,Y_test)))
dataset.plot(kind='box',sharex=False,sharey=False)
nb_model.predict([[5.1,3.5,1.4,0.2]])
#dataset.hist(edgecolor='black',linewidth=1.2)

#sns.violinplot(data=dataset,x="Species", y="PetalLengthCm")
nb_model.predict([[4.6,3.6,1,0.2]])
nb_model.predict([[6,3.4,5,1.6]])
nb_model.predict([[7.7,3,6.1,2.3]])
nb_model.predict([[7.2,3,5.8,0]])
nb_model.predict([[6.3,3.3,4.7,1.6]])
nb_model.predict([[6.4,2.8,5.6,2.1]])
nb_model.predict([[5.2,4.1,1.5,0.1]])




nb_model.predict([[5.7,2.7,3.9,1.4]])



