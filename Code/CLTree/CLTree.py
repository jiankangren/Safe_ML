import numpy as np
from Score_Func import *
import copy
import itertools
import pudb
from plot import *

class CTree(object):
    '''
    -----------------------------------
    Robust Classification Tree
    -----------------------------------
    Parameter:
    - data: training data
    - n_feature: number of features
    - feature_limit: the range for each feature {0: (-3, 10),1:(3,4),2....} :
    -----------------------------------
    ''' 
    def __init__(self, data):
        super(CTree, self).__init__()
        self.data = np.array(data)

    @property
    def n_feature(self): 
        return self.data.shape[1]-1  
    @property
    def feature_limit(self): 
        ranges={}
        for j in xrange(0,self.n_feature):
            ranges[j]=(min(self.data[:,j]), max(self.data[:,j]))
        return ranges
        F={}
        for i in xrange(0,self.n_feature):
            key1=(i,0)
            key2=(i,1)
            F[key1]=self.ranges[i][1]
            F[key2]=self.ranges[i][0]
        return F


    def data_split(self, data,index, value):
        '''
        -----------------------------------------------------------
        Split a dataset based on an attribute and an attribute value
        -----------------------------------------------------------
        Input: 

        - index: split index
        - value: split data by <=value
        
        Return: 
        - two splited array of data 

        '''
        left_index=data[:,index]<=value
        right_index=np.logical_not(left_index)
        return data[left_index], data[right_index]
 

    def best_split(self,data,score_func):
        '''
        ------------------------------
        Greedy Seach of the best split
        ------------------------------
        Input: 

        - score_func: a score function 

        Return:

        {'index':b_index,'value':b_value, 'groups':b_groups} 


        '''
        assert issubclass(score_func, ScoreFunc)
        if not isinstance (data,np.ndarray):
            raise AttributeError('Dataset is not numpy dnarray')
        class_values = list(set(data[:,-1]))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        key=data[:,-1]!=999
        sdata=data[key]
        for index in range(len(data[0])-1): 
            for row in sdata:
                split_value=row[index]
                groups = self.data_split(data,index, split_value)
                score = score_func.score(groups, class_values)
                if score < b_score:
                    b_index, b_value, b_groups,b_score = index, row[index], groups,score
        return {'index':b_index,'value':b_value, 'groups':b_groups}



    def loop_split(self, node, max_depth, min_size, depth,score_func,purity_threshold=0.95):
        '''
        -----------------------------------------
        Build tree node (dictionary) recursively.
        -----------------------------------------

        Input:
        - node: a tree node dictionary {'index':split index, 'Prob': histgram of classes,
        Purity: the maximum percentage of a class wihtin node, 'left/right': child trees}
        - max_depth  : for  the  tree
        - min_size:  the minimal number of instances a node does not need to further split and the node termiates
        - purity_threshold:  if the purity exceeds the threshold, the node does not need to further split and the node termiates

        '''
         
        left, right = node['groups']
        del(node['groups'])
        PL=self.check_purity(left)
        PR=self.check_purity(right)
        if PL==None or PR==None:  # if one group is empty, then terminates
            P=self.check_purity(np.vstack((left,right)))
            node['Prob']=P[1]
            node['Purity']=P[0]
            return 
      
        if depth >= max_depth: 
            print 'depth', depth, 'is greater than','max_depth',max_depth
            node['left']={}
            node['left']['Prob']=PL[1]
            node['left']['Purity']=PL[0]
            node['right'] ={}
            node['right']['Prob']=PR[1]
            node['right']['Purity']=PR[0]
            return
        
        if PL[0]>purity_threshold or len(left) <= min_size: # no further split and terminate
            node['left']={}
            node['left']['Prob']=PL[1]
            node['left']['Purity']=PL[0]
        else:
            node['left'] = self.best_split(left,score_func)
            self.loop_split(node['left'], max_depth, min_size, depth+1,score_func)

        if PR[0]>purity_threshold or len(right) <= min_size: # no further split and terminate
            node['right']={}
            node['right']['Prob']=PR[1]
            node['right']['Purity']=PR[0]
        else:
            node['right'] = self.best_split(right,score_func)
            self.loop_split(node['right'], max_depth, min_size, depth+1,score_func)
  
    def check_purity(self,data):
        '''
        -----------------------------------------
        Check purity  (maximum percentage of a node) of a node.
        -----------------------------------------
        
        Return

        - Prob: {class1:size, class2:size...}
        - :    the class with the largets percentge 

        '''
 
        if len(data)==0:
            return None
        Prob={}
        class_values = list(set(data[:,-1]))
        size=float(len(data))
        for c in class_values:
            keyc=(data[:,-1]==c)
            Prob[c]=len(data[keyc])
        return  round(Prob[max(Prob, key=Prob.get)]/size,2), Prob

    def build_tree(self, max_depth, min_size,purity_threshold=0.95):
        '''
        ----------------------
        Build the tree.
        ----------------------
        Input: 
        - max_depth: maximum depth of the tree
        - min_size: minimal size of a node to stop further spliting
        - purity_threshold: minimum purity of a node to stop further spliting
        
        Return:
        -root: the  dictionary of the tree.
        '''
        data=self.data
        purity,prob=self.check_purity(data)
        if purity>purity_threshold:
            return {'Prob':prob,'Purity':purity}
        else:
            root = self.best_split(data,GiniFunc)
            self.loop_split(root, max_depth, min_size, 1,GiniFunc)
            return root

    def predict(self,node,row):
        '''
        ----------------------
        Make a prediction
        ----------------------
        Input
    
        node: tree node
        row:  a data point 
        ----------------------
        Return
        c:  class of prediction
        P:  histgram of leaf node
    
        '''
    
        if 'index' in node.keys():
            if row[node['index']] < node['value']:
                if isinstance(node['left'], dict):
                    return self.predict(node['left'], row)
                else:
                    return node['left']
            else:
                if isinstance(node['right'], dict):
                    return self.predict(node['right'], row)
                else:
                    return node['right']
        else:
            P=node['Prob']
            c=max(P ,key=P.get)
            return c,P


    def extract_rules(self,node,min_size=5,min_purity=0.9):
        '''
        -----------------------------
        Extract Rules from a root node
        -----------------------------
        Input:
        - node: tree node
        - min_size: the minimum size of instances in a node so that the rule is extracted
        - min_purity: the minimum purity of a node so that the rule is extracted

        Return:
        - Rules:[{rule}]
        - rule: {'index_logic':{(0,0):1,(0,1):2....,}, 'Prob':{class1:32,..}, 'Purity':0.9,'class':class1}


        '''
    
        Rules=[]
        def extract(node,rules=[{'index_logic':np.array([])}]):
            '''
            ----------------------------------------------------------
            Extract all the rules from tree node and add them in rules
            ----------------------------------------------------------

            '''
            if 'index' in node.keys():
                Lrules= copy.deepcopy(rules[:])
                Rrules= copy.deepcopy(rules[:])
                for rule in Lrules:
                    rule['index_logic'].append({(node['index'],0):node['value']})
                if 'left' in node.keys():
                    extract(node['left'],Lrules)
                for rule in Rrules:
                    rule['index_logic'].append({(node['index'],1):node['value']})
                if 'right' in node.keys():
                    extract(node['right'],Rrules)
            else:
                for rule in rules:
                    rule['Prob']=node['Prob']
                    rule['Purity']=node['Purity']
                    rule['class']=max(node['Prob'],key=node['Prob'].get)
                Rules.append(rules)
        extract(node,rules=[{'index_logic':[]}])
        '''
        ----------------------------------------------------
        remove rules that without enough samples and purity
        ----------------------------------------------------

        '''
        cRules=[]
        for rule in Rules:
            key=rule[0]['Prob'].keys()[0]
            if rule[0]['Prob'][key]>min_size and rule[0]['Purity']>min_purity:
                cRules.append(rule[0])
        Rules=cRules
       
        def clean_rule():
            '''
            ------------------------------------------------------
            remove redundant rules
            Example: 
            'index 0<4'  and  'index 0<6'  is merged to 'index 0<4'
            ------------------------------------------------------
            '''
            for rule in Rules:
                index_logic=rule['index_logic']
                cil={}
                for il in index_logic:
                    key=il.keys()[0]
                    if key not in cil.keys():
                        cil[key]=il[key]
                    else:
                        if key[1]==0:
                            cil[key]=min(cil[key],il[key])
                        elif key[1]==1:
                            cil[key]=max(cil[key],il[key])
                rule['index_logic']=cil
        clean_rule()
        def add_limit():
            '''
            ------------------------------------------------------
            Add feature limit to each rule if needed
            ------------------------------------------------------
            '''
            F=self.feature_limit
            Fkey=F.keys() #[0,1,2,3]
            for rule in Rules:
                ig=rule['index_logic']
                for key in Fkey:
                    if (key, 0) not in ig.keys():
                        ig[(key, 0)]=F[key][1]
                    if (key, 1) not in ig.keys():
                        ig[(key, 1)]=F[key][0]
        add_limit()
        return Rules




    








    
        



from sklearn.datasets.samples_generator import make_blobs
def samples_generator(n_samples):
    '''
    ------------------------------------------------------
    Generate samples
    ------------------------------------------------------
    '''

    X, y = make_blobs(n_samples, centers=2, n_features=2,random_state=0)
    X=np.array(X)
    y=np.array([y])
    y=y.reshape(y.shape[1],y.shape[0])
    data=np.concatenate((X, y), axis=1)


    a=np.linspace(min(data[:,0]),max(data[:,0]),10,endpoint=True)
    b=np.linspace(min(data[:,1]),max(data[:,1]),10,endpoint=True)
    s=np.array(list(itertools.product(a, b)))
    y=[[999]]*len(s)# 999 represents the virtual class
    ds=np.concatenate((s, y), axis=1)
    data=np.vstack((data ,ds))
    return data







if __name__=='__main__':
    data = np.array([
    [-2.571244718,4.784783929,0],
    [-3.571244718,5.784783929,0],
    [-3.771244718,1.784783929,1],
    [-2.771244718,1.784783929,1],
    [2.771244718,1.784783929,0],
    [1.728571309,1.169761413,0],
    [3.678319846,2.81281357,0],
    [3.961043357,2.61995032,0],
    [2.999208922,2.209014212,0],
    [7.497545867,3.162953546,1],
    [9.00220326,3.339047188,1],
    [7.444542326,0.476683375,1],
    [10.12493903,3.234550982,1],
    [6.642287351,3.319983761,1]])



    data=samples_generator(200)
    
    # print data
    ctree=CTree(data)

    # print ctree.feature_limit
    # print ctree.n_feature
    # print ctree.data_split(data,0,0)
    # root =ctree.best_split(data,GiniFunc)
    tree =ctree.build_tree(max_depth= 15, min_size= 10,purity_threshold=0.95)
    print tree
    
    
    rules=ctree.extract_rules(tree)
    for i in rules:
        print i
    plot_scatter(data)
    plot_all_lg(rules)



    plt.show()

