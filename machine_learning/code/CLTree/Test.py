from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from RandomForest import * 
from base import *
from DecisionTree import *
from math import floor,ceil
def time_rf():
    n=2000
    X, y = make_classification(n_samples=n,n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=2)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    moon=make_moons(n_samples=n,noise=0.3, random_state=0)
    circle=make_circles(n_samples=n,noise=0.2, factor=0.5, random_state=1)  
    X,y=    circle
    h=0.05
    x1_min, x1_max = X.T[0].min() - .1, X.T[0].max() + .1
    x2_min, x2_max = X.T[1].min() - .1, X.T[1].max() + .1
    xx, yy = np.meshgrid(np.arange(x1_min-1, x1_max+1, h),np.arange(x2_min-1, x2_max+1, h))
    rf=RandomForestClassifier(nb_trees=20, pec_samples=0.4, max_workers=6,\
        criterion='gini',  min_samples_leaf=1, max_depth=30,gain_ratio_threshold=0.01,\
        max_feature_p=1,max_try_p=0.5)
    rf.fit(X)
    Z= rf.predict(np.c_[xx.ravel(), yy.ravel()],0.01)
    Z=Z.reshape(xx.shape)
    ax = plt.subplot(1, 1, 1)
    cm = plt.cm.tab20c
    ax.set_title("")
    ax.scatter(X.T[0], X.T[1], c=y, cmap=cm,edgecolors='k')
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xticks(range(int(floor(x1_min)),int(ceil(x1_max))))
    ax.set_yticks(range(int(floor(x2_min)),int(ceil(x2_max))))
    cntr1 = ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    cbar0 = plt.colorbar( cntr1,)
    plt.tight_layout()
    plt.show()





