from RandomForest import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# mydata=pd.read_csv('./data/2.csv').values
# X=np.array(mydata[:,0:2])
# y=mydata[:,2]

# X_train, X_test, y_train, y_test = \
#         train_test_split(X, y, test_size=.25, random_state=42)   
# mlp = MLPClassifier(hidden_layer_sizes=(100,100,), max_iter=50, alpha=1e-4,
#                     solver='adam', verbose=10, tol=1e-5, random_state=1,
#                     learning_rate_init=.1)
# mlp.fit(X_train, y_train)

# print("Training set score: %f" % mlp.score(X_train, y_train))

# print("Test set score: %f" % mlp.score(X_test, y_test))
# label=mlp.classes_
# y_pred= np.array(mlp.predict_proba(X))
# y_pred_index=np.argmax(y_pred, axis=1)
# y_h= np.array([ [label[i],yt,p[i]] for i,p,yt in zip(y_pred_index,y_pred,y)])    
# index_wrong=(y_h.T[0]!=y_h.T[1])
# print y_h[index_wrong]

# rf=RandomForestClassifier(nb_trees=20, pec_samples=0.4, max_workers=4,\
#     criterion='gini', min_samples_leaf=1, max_depth=30,gain_ratio_threshold=0.001)
# rf.fit(X)
# Z= rf.predict(X[index_wrong],0.2)
# print Z


x=np.array([1,2,3,4])
a=np.array([2,3,4,5])
y=np.array([3,4,5,6])


# def f():
#     return np.prod(y-x)

# print f()

sat0=(x<=a)
sat1=(a<=y)
z=np.array([sat0,sat1])
print np.all(z)