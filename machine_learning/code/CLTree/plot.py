from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from math import floor,ceil
import numpy as np
from matplotlib.lines import Line2D
from six import iteritems

markers=['.','+','x','_',',','|','1','2','3','4','8','>','<']
def plot_scatter(X,y,label,x_label='',y_label=''):
    global markers
    figure = plt.figure(figsize=(5, 5))
    h=0.05
    x1_min, x1_max = X.T[0].min() - .1, X.T[0].max() +.1
    x2_min, x2_max =X.T[1].min() - .1, X.T[1].max() + .1
    # xx, yy = np.meshgrid(np.arange(x1_min-1, x1_max+1, h),np.arange(x2_min-1, x2_max+1, h))
    cmap1 = plt.cm.tab20c
    ax = plt.subplot(1, 1, 1)
    ax.set_title(" ")
    ax.set_ylim(x2_min, x2_max)
    ax.set_xticks(range(int(floor(x1_min)),int(ceil(x1_max))))
    ax.set_yticks(range(int(floor(x2_min)),int(ceil(x2_max))))
    ax.set_xlabel(x_label,size=20)
    ax.set_ylabel(y_label,size=20) 
    for j in range(0,len(label)):
        index=(y==label[j])
        ax.scatter(X[index].T[0], X[index].T[1],marker=markers[j],cmap=cmap1,label=label[j])     
        ax.set_xlim(x1_min, x1_max)
    plt.legend()
    plt.tight_layout()
    plt.show()








if __name__ == '__main__':

    n=2000
    X, y = make_classification(n_samples=n,n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=2)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    moon=make_moons(n_samples=n,noise=0.3, random_state=0)
    circle=make_circles(n_samples=n,noise=0.2, factor=0.5, random_state=1)  
    X,y=  circle
    plot_scatter(X,y,[0,1])
    
    
    
    



