
# coding: utf-8

# In[28]:


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



# In[29]:


#names=["sepal_length" , "sepal_width","petal_length","petal_width","class"]
#dataset=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",names=names)
dataset=pd.read_csv("D.csv")


# In[30]:


dataset.shape


# In[31]:


dataset.tail


# In[32]:


dataset.head


# In[33]:


dataset.describe


# In[34]:


dataset.describe()


# In[35]:


input_columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
#input_columns=[:,0:4]


# In[36]:


output_columns=["Species"]
#output_columns=[:,4]


# In[37]:


#X=dataset.iloc[: ,:-1].values
#print (X)
#Y=dataset.iloc[:,-1].values
#print (Y)
X=dataset[input_columns].values
Y=dataset[output_columns].values


# In[38]:


#split_test_size=0.20


# In[39]:


X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.30,random_state=5)


# In[40]:


vatchala={
        'Support':SVC(),
          'Rand':RandomForestClassifier(),
          'Tree':DecisionTreeClassifier(),
          #'Naive':GaussianNB(),
        'AdaBoost':AdaBoostClassifier()
         }


# In[41]:


for N,V in vatchala.items():
    nb_model=V
    nb_model.fit(X_train,Y_train)
    nb_predict_train=nb_model.predict(X_train)
    nb_predict_test=nb_model.predict(X_test)
    print(N,"Accuracy:{0:.4f}".format(metrics.accuracy_score(nb_predict_test,Y_test)))


# In[42]:


dataset.plot(kind='box',sharex=False,sharey=False)


# In[43]:


nb_model.predict([[5.1,3.5,1.4,0.2]])


# In[44]:


#dataset.hist(edgecolor='black',linewidth=1.2)


# In[45]:


#sns.violinplot(data=dataset,x="Species", y="PetalLengthCm")


# In[46]:


nb_model.predict([[4.6,3.6,1,0.2]])


# In[47]:


nb_model.predict([[6,3.4,5,1.6]])


# In[48]:


nb_model.predict([[7.7,3,6.1,2.3]])


# In[49]:


nb_model.predict([[7.2,3,5.8,0]])


# In[50]:


nb_model.predict([[6.3,3.3,4.7,1.6]])


# In[51]:


nb_model.predict([[6.4,2.8,5.6,2.1]])


# In[56]:


nb_model.predict([[5.2,4.1,1.5,0.1]])


# In[53]:


nb_model.predict([[5.7,2.7,3.9,1.4]])


# In[54]:


nb_model.predict([[1,5.1,3.5,1.4]])


# In[55]:





# In[60]:


nb_model.predict([[5.1,3.8,1.6,0.2]])

