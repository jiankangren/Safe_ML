import pydot
""" 
-----------------------------
draw decison tree using pydot
-----------------------------
"""

def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)

def plot(node, parent=None):
    graph = pydot.Dot(graph_type='graph')
    draw('index0', 'index1')
    draw('index0', 'index2')
    draw('index0', 'index3')
    graph.write_png('example1_graph.png')
    Image('example1_graph.png')



'''
---------------------------------------------
plot the robust DT classification for 2d data
---------------------------------------------
'''

import matplotlib.pyplot as plt
'''
---------------------------------------------
plot the rectangle based on rule   a<x<b  c<y<d 
---------------------------------------------
'''
def plot_rect(lg,clr):
    """
    Input:
    ----------
    lg: {(0, 1): 3, (1, 0): 6,
         (0, 0): 4, (1, 1): 2}
    ect_color: (str) of color

    """
    a=lg[(0,0)]
    b=lg[(0,1)]
    c=lg[(1,0)]
    d=lg[(1,1)]
    x1, y1 = [a, a], [d, c]
    x2, y2 = [b, b], [d, c]
    x3, y3 = [a, b], [c, c]
    x4, y4 = [a, b], [d, d]
    plt.plot(x1, y1,x2,y2, x3,y3,x4,y4,marker = '.',alpha=0.5,color=clr)

'''
--------------------------------
plot the rectangle for all rules
--------------------------------
'''
def plot_all_lg(rules):
    """
    Input:
    ----------
    rules [{(0, 1): 3, (1, 0): 6,(0, 0): 4, (1, 1): 2}, ...]

    """
    color_list=['m', 'y', 'k', 'w','b', 'g', 'r', 'c']
    class_to_color={}
    # the number of classes cannot exceed  color list
    for rule in rules:
        lg=rule['index_logic']   
        c=rule['class']
        if c not in class_to_color.keys():
            class_to_color[c]=color_list.pop()
        if c!=999:
            plot_rect(lg, class_to_color[c]) 
 
'''
-----------------------------------
plot scatter for all points in data
-----------------------------------
'''   

def plot_scatter(data,NUM_COLORS = 5): #2d scatter
    
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    class_values=set(data[:,-1])
    for c in class_values:
        key =(data[:,-1]==c)
        C=data[key]
        x,y=C[:,0],C[:,1]
        area = [20]*len(x)  # 0 to 15 point radii
        if c!=999:
            plt.scatter(x, y, s=area,  alpha=0.5)
