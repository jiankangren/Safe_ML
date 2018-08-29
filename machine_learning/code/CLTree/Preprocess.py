import numpy as np
import pandas as pd
from math import ceil, floor
from RandomForest import *
from sklearn.model_selection import train_test_split

file1=['./data/robot/','2.csv']


def store_output(X,filename):
    file='output/'+filename+'.npy'
    np.save(file, X)    # .npy extension is added if not given

def read_output(filename):
    file='output/'+filename+'.npy'
    X = np.load(file)
    return np.array(X)


def read_data(path):
    X1 = np.load(path+'X1.npy')
    y1= np.load(path+'y1.npy')
    X2 = np.load(path+'X2.npy')
    y2= np.load(path+'y2.npy')
    return np.array(X1),np.array(X2),np.array(y1),np.array(y2)

'''use the top lowest p% density  data as test
'''
def my_split(file,p=0.03,test_size=0.5):
    rf=RandomForestClassifier(nb_trees=25, pec_samples=0.4, max_workers=8,\
        criterion='gini', min_samples_leaf=1, max_depth=25,gain_ratio_threshold=0.001,\
                               max_feature_p=1,max_try_p=1)
    
    path=file[0]
    filename=path+file[1]
    mydata=pd.read_csv(filename).values
    X=np.array(mydata[:,0:2])
    y=mydata[:,2]
    rf.fit(X)
    Z=rf.predict(X,0)
    Z.sort()
    L=int(floor(p*len(X)))
    threshold=Z[L]
    Z=rf.predict(X,0)
    index_rare= Z<threshold
    index_normal=Z>=threshold
    X_train, X_test, y_train, y_test = \
        train_test_split(X[index_normal],y[index_normal], test_size=0.3, random_state=1)   
    X_test=np.concatenate((X_test,X[index_rare]), axis=0)
    y_test=np.concatenate((y_test,y[index_rare]), axis=0)
    np.save(path+'X1.npy', X_train) 
    np.save(path+'y1.npy', y_train) 
    np.save(path+'X2.npy', X_test) 
    np.save(path+'y2.npy', y_test) 


# my_split(file1)






# X1,y1,X2,y2=read_data(file1[0])
# # print y1[0:10]

# from plot import *
# plot_scatter(X1,y1)
# plot_scatter(X2,y2)






