from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from RandomForest import * 
from base import *
from DecisionTree import *
from math import floor,ceil
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared,ConstantKernel 

from sklearn.svm import SVR



file1=['./data/robot/2/','2.csv']
file2=['./data/SARCOS/','train.mat']




__SARCOS__=True
if __SARCOS__:
    file=file2
    X_train, X_test, y_train, y_test = read_data(file[0])
    X=np.concatenate((X_train,X_test), axis=0)
    y=np.concatenate((y_train,y_test), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.9, random_state=1)   


    """
    ===================================================================================
    Train CLTree
    """
    # tree=CLTree(criterion='gini',min_samples_leaf=5, max_depth=80,gain_ratio_threshold=0.00001)
    # tree.fit(X_train,1,0.5)
    # save(tree,file[0]+'tree.pkl')
    
    """
    Load Tree and Predict
    """

    tree=load(file[0]+'tree.pkl')
    yt=tree.predict(X_train)
    yp=tree.predict(X_test)

    yt.sort()
    k=int(floor(len(yt)*0.01))
    threshold=yt[k]
    print 'Threshold:', threshold
    # ax = plt.subplot(2, 2, 1)
    # plt.hist(tree.density,bins=25,density=True, histtype='step',cumulative=True)
    # ax = plt.subplot(2, 2, 2)
    # plt.hist(tree.density_X,bins=25,density=True, histtype='step',cumulative=True)




    # ax = plt.subplot(2, 2, 3)
    # plt.hist(yt[yt>0],bins=25,density=True, histtype='step',cumulative=True)
    # ax = plt.subplot(2, 2, 4)
    # plt.hist(yp[yp>0],bins=25,density=True, histtype='step',cumulative=True)
    # plt.tight_layout()
    # plt.show()


    inds=[yp<0.2,np.logical_and(yp>=0.2,yp<0.4), np.logical_and(yp>=0.4,yp<0.6),\
         np.logical_and(yp>=0.6,yp<0.8),np.logical_and(yp>=0.9,yp< threshold),np.logical_and(yp>=threshold,yp<1)]
    # inds=[yp<threshold,yp>=threshold]
   



    """
    ===================================================================================
    Train MLP and Predict
    List MSE
    """
    # from sklearn.neural_network  import MLPRegressor
    # mlp = MLPRegressor(hidden_layer_sizes=(200,150,100,), max_iter=500, alpha=1e-8,
    #                 solver='adam', verbose=10, tol=1e-5, random_state=1, batch_size='auto',
    #                 learning_rate_init=.01)
    # mlp.fit(X_train, y_train)
    # print("Training set score: %f" % mlp.score(X_train, y_train),len(y_train))
    # y_pred_train=mlp.predict(X_train)
    # y_pred_test=mlp.predict(X_test)


    # print mean_squared_error(y_train, y_pred_train)
    # print mean_squared_error(y_test, y_pred_test)

    # ind=np.argsort(yp)
    # R=np.square(y_test[ind]-y_pred_test[ind])
    # save( R,file[0]+'mlpbox')


    """
    ===================================================================================
    Train SVM  and Predict
    List MSE and STD
    """
    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.01)
    # svr_rbf.fit(X_train, y_train)
    # print("Training set score: %f" % svr_rbf.score(X_train, y_train),len(y_train))
    # y_pred_train=svr_rbf.predict(X_train)
    # y_pred_test=svr_rbf.predict(X_test)
    
    # print mean_squared_error(y_train, y_pred_train)
    # print mean_squared_error(y_test, y_pred_test)
    

    # ind=np.argsort(yp)
    
    # # R=[]
    # # for ind in inds:
    #     # print mean_squared_error(y_test[ind], y_pred_test[ind]) ,len(y_test[ind])
    #     # r=np.square(y_test[ind]-y_pred_test[ind])
    #     # R.append(r)
    # R=np.square(y_test[ind]-y_pred_test[ind])
    # save( R,file[0]+'svmbox')

    """
    ===================================================================================
    Train GP and Predict
    List MSE and STD
    """
    X_train, X_test, y_train, y_test = read_data(file[0])
    X=np.concatenate((X_train,X_test), axis=0)
    y=np.concatenate((y_train,y_test), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.97, random_state=1)   
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    gp.fit(X_train, y_train)
    y_pred_train, sigma_train = gp.predict(X_train, return_std=True)
    y_pred_test, sigma_test = gp.predict(X_test, return_std=True)
    print mean_squared_error(y_train, y_pred_train)
    print mean_squared_error(y_test, y_pred_test)

    tree=load(file[0]+'tree.pkl')
    yt=tree.predict(X_train)
    yp=tree.predict(X_test)
    yt.sort()
    k=int(floor(len(yt)*0.01))
    # threshold=yt[k]
    # print 'Threshold:', threshold
    # inds=[yp<0.2,np.logical_and(yp>=0.2,yp<0.4), np.logical_and(yp>=0.4,yp<0.6),\
    #      np.logical_and(yp>=0.6,yp<0.8),np.logical_and(yp>=0.9,yp< threshold),np.logical_and(yp>= threshold,yp<1)]
    


    # for ind in inds:
    #     print mean_squared_error(y_test[ind], y_pred_test[ind]) ,len(y_test[ind]),np.mean(sigma_test[ind])
    ind=np.argsort(yp)
    R=np.square(y_test[ind]-y_pred_test[ind])
    save( R,file[0]+'gpbox')

    R=np.square(sigma_test[ind])
    save( R,file[0]+'gpboxstd')



