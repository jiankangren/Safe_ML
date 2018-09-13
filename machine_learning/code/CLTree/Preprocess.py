import numpy as np
import pandas as pd
from math import ceil, floor
from RandomForest import *
from sklearn.model_selection import train_test_split
from scipy import io
import pickle
from sklearn.neural_network import MLPClassifier

"""
-------------------------------------------------------------------       
Data Set Initialize
-------------------------------------------------------------------
"""
def SARCOS():
    mat = io.loadmat('data/SARCOS/train.mat')
    matrix=mat['sarcos_inv']
    data_train=pd.DataFrame(matrix).values
    mat = io.loadmat('data/SARCOS/test.mat')
    matrix=mat['sarcos_inv_test']
    data_test=pd.DataFrame(matrix).values
    data=np.concatenate((data_train,data_test),axis=0)
    return data



"""
-------------------------------------------------------------------
https://www.kaggle.com/uciml/wall-following-robot#sensor_readings_2.csv
"""
def WALL_FOLLOW_ROBOT(file):
    data=pd.read_csv(file[0]+file[1]).values
    return data

"""
-------------------------------------------------------------------

===================================================================
"""





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


'''use the top max/min feature as test
'''
def data_split(data,file,p=0.2):
    path=file[0]
    n_feature=len(data.T)-1
    L=int(floor(len(data)*p))
    data_test=None
    X=data[:,0:n_feature]
    y=data[:,n_feature]
    mlp = MLPClassifier(hidden_layer_sizes=(200,150,100,), max_iter=200, alpha=1e-6,\
                    solver='adam', verbose=10, tol=1e-6, random_state=1,learning_rate_init=.05)
    mlp.fit(X,y)
    score=mlp.score(X, y)
    print 'Score:',score

    label=mlp.classes_
    y_pred= np.array(mlp.predict_proba(X))
    y_pred_index=np.argmax(y_pred, axis=1)
    yh= np.array([ [label[i],y_true] for i,y_true in zip(y_pred_index,y)])    
    ymlp= np.array([ float(prob[i]) for i, prob in zip(y_pred_index,y_pred)])
    index=np.argsort(ymlp)
    index1=index[0:L]
    index2=index[L:]
    np.save(path+'X1.npy',X[index2]) 
    np.save(path+'y1.npy',y[index2]) 
    np.save(path+'X2.npy',X[index1]) 
    np.save(path+'y2.npy',y[index1]) 







"""
-------------------------------------------------------------------
Store and Load Model
-------------------------------------------------------------------
"""

def save_model(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_model(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)







if __name__=='__main__':
    """WALL_FOLLOW_ROBOT
    """
    _WALL_FOLLOW_ROBOT=True
    if _WALL_FOLLOW_ROBOT:
     
        file1=['./data/robot/2/','2.csv']
        file1a=['./data/robot/4/','4.csv']
        file1b=['./data/robot/24/','24.csv']
        
        data=WALL_FOLLOW_ROBOT(file1)
        data_split(data,file1,0.5)

        data=WALL_FOLLOW_ROBOT(file1a)
        data_split(data,file1a,0.5)

        data=WALL_FOLLOW_ROBOT(file1b)
        data_split(data,file1b,0.5)

    """SARCOS
    """
    __SARCOS__=False#True
    if __SARCOS__:
        file2=['./data/SARCOS/','train.mat']
        data=SARCOS()
        data_split(data,file2,0.5)
    












