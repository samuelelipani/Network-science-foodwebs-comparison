import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
from TrophicLevel import *
from NodeScores import *
from Degree_PageRank_HITS import *


############################ Graph ###########################



def create_fixed_positions(A,          # adj matrix
                           ybasal,     # y coordinate of all basal species
                           ytop,       # y coordinate of all top species
                           spacing     # x distance between two near basal or top species
                           ):
    
    """
    assigns to the basal and top nodes the centered nospaced (x,y) coordinate-couples at the botton and top
        Args
            A       :  adj matrix
            ybasal  :  y coordinate which will be assigned to all basal species
            ytop    :  y coordinate which will be assigned to all top species
            spacing :  spacing on the lattice between nodes of the same type (basal or top)

        Returns
            fixed_positions :  dictionary of key value pairs.
                                keys = top and basal node index (i)
                                value = (x,y) positions assigned to each of this nodes

    """
    
    basal = basal_species(A)  # basal mask
    top = top_species(A)      # top mask
    
    fixed_positions = {}
    n = 0
    m = 0
    
    for i in range( len(top) ):  # over all node indexes
            
        if(top[i]==1):
            x = int( (m+1)/2 )*(-1)**m              # assigns x positions 0,1,-1,2,-2,3,-3 and so and so forth to m = 0,1,2,3,...
            fixed_positions[i] = (x*spacing,ytop)   # rescales with chosen spacing and assign to corresponding node
            m = m+1

        if(basal[i]==1):
            # n=0 x=0, n=1 x=-1, n=2 x=1, n=2 x=-2, ... 
            x = int( (n+1)/2 )*(-1)**n              # same as before
            fixed_positions[i] = (x*spacing,ybasal)    
            n = n+1
 
    return fixed_positions




def dir_graph(Adf,                                    # adj matrix as dataframe with labels, or np array. In that case put df = False !
              node_colors,                            # vector of scores, one for each node
              colormap_label,                         # needed because there are no fixed color choices
              node_size, 
              width,
              df=True,                                # handles non df input
              seed = 4020,                            # not relevant if basal/top nodes are present
              correction = True,                      # manual modification of nodes height
              modified_heights = "trophic2",          # "trophic2", "influence" or "hubs_pagerank"
              ybasal=-1, ytop=1, spacing=0.1,         # parameters of position of fixed nodes
              edge_color='gray', cmap=plt.cm.gnuplot, # colormap parameters
              top_labels = False,                     # if true names of top nodes are displayed
              basal_labels = False,                   # if true names of basal nodes are displayed
              labels_font_size = 8,                   # names font size
              top_label_raise = 0.1,                  # how much to raise labels of top species
              basal_label_lower = 0.1,                # how much to lower labels of basal species
              fig1 = None, ax1 = None,                # figure and axis on which we want to plot
              colorbarway = False, 
              arrows = False                    # how we want to display the colorbar
              ): 
    
    make_new_figure = True if ((fig1 is None) or (ax1 is None)) else False

    A = Adf
    if(df):
        A = Adf.values
    
    ###### Spring simulation ######
    
    # we simulate A as an "undirected" spring system, otherwise every node is rejected
    # by input edges, which is not good.
    G = nx.from_numpy_matrix(A, create_using=nx.Graph)
    
    # fixing position of top and basal nodes to impose structure
    fixed_positions = create_fixed_positions(A,ybasal,ytop,spacing)
    fixed_nodes = fixed_positions.keys()
    
    # dynamic simulation
    if(len(fixed_nodes)): 
        # seed is not relevant thanks to contrained positions
        pos = nx.spring_layout(G,pos=fixed_positions, fixed = fixed_nodes)    
    else: 
        pos = nx.spring_layout(G, seed=seed)

    plt.figure(figsize=(8, 6))
    
    # this is needed to have arrows
    if arrows == True:
        G_dir = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    else : G_dir = nx.from_numpy_matrix(A, create_using=nx.Graph)
    
    # Now we have our graph with basal species on the bottom layer and top in the top layer
    # but there is a problem. Because of the spring dynamic, semi-top predator which hunts a lot
    # of basal species are veery close to basal layer!
    #
    # In order to correct this we can fix the x position of the nodes, and set the y coordinate
    # equal for example to the trophic level squared!
    if(correction):
        
        if (modified_heights == "trophic2"):
            new_heights = compute_trophic_levels(A,False)**2
            
        elif (modified_heights == "influence"):
            new_heights = score_distribution(A,"influence")
        
        elif (modified_heights == "hubs_pagerank"):
            A = A.T 
            # normalizing A columns to sum 1
            sums = A.sum(axis = 0) + 2**(-32)    
            M = A/sums                           
            new_heights = PageRank(M,c=0.7)
            
        else:
            print("Error, unknown height correction used!")
            return 0
            
        #new_heights = fw.PageRank(A.T)
        
        for i in range(len(pos)):
            new_pos = pos[i]
            new_pos[1] = new_heights[i]
            
            pos[i] = new_pos
    
    
    ################ GRAPH #############
    
    if make_new_figure: fig1, ax1 = plt.subplots(figsize=(10, 10))

    nx.draw(G_dir, pos, node_color = node_colors, width=width, 
            node_size=node_size, edge_color=edge_color, cmap=cmap, ax=ax1)   
        
    ####### Nodes name #######
    
    names = []
    if(df):
        names = list(Adf.columns.values)
    
    labels = {} 
    
    
    if(top_labels):
        top = np.where( top_species(A)==1 )[0] 
        
        for node in G_dir.nodes():
            if node in top:
                #set the node name as the key and the label as its value 
                labels[node] = names[node]
                
        description = nx.draw_networkx_labels(G_dir,pos,labels,font_size=labels_font_size,font_color='k',
                                              verticalalignment="bottom")
        
        # modifying labels
        for node, t in description.items():
            # rotating labels
            t.set_rotation(90)
            
            # manually raising y value of labels, in this way they won't overlap with nodes
            position = list(t.get_position())
            position[1] = position[1] + top_label_raise
            t.set_position(position) 
            
            t.set_clip_on(False)   
            
                
    labels = {} 
    
    if(basal_labels):
        basal = np.where( basal_species(A)==1 )[0] 
        
        for node in G_dir.nodes():
            if node in basal:
                #set the node name as the key and the label as its value 
                labels[node] = names[node]

        description = nx.draw_networkx_labels(G_dir,pos,labels,font_size=labels_font_size,font_color='k',
                                              verticalalignment="top")
        
        # modifying labels
        for node, t in description.items():
            # rotating label
            t.set_rotation(90)
            
            # manually lowering y value of labels, in this way they won't overlap with nodes
            position = list(t.get_position())
            position[1] = position[1] - basal_label_lower
            t.set_position(position) 
            
            t.set_clip_on(False)
            
            
    ax1.set_axis_on()
    
    if(correction):
        
        if (modified_heights == "trophic2"):
            ax1.set_ylabel("Trophic level squared",fontsize=20)
            
        elif (modified_heights == "influence"):
            ax1.set_ylabel("Influence score",fontsize=20)
        
        elif (modified_heights == "hubs_pagerank"):
            ax1.set_ylabel("Pagerank hubs",fontsize=20)
    
    
   
    ax1.tick_params(left=True,labelleft=True,which='major', labelsize=15)
    ax1.grid(axis="y",color='gray', linestyle='--', linewidth=1, alpha=0.4) 
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    
    ####### Colorbar ####### 
    if colorbarway == False: # old way of making the colorbar
        fig2, ax2 = plt.subplots(figsize=(11, 0.5))
        fig2.subplots_adjust(bottom=0.5)
    
        norm=plt.Normalize(vmin = np.min(node_colors), vmax=np.max(node_colors))
    
        cbar = fig2.colorbar( mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                             cax=ax2, extend='max', orientation='horizontal')
    
        cbar.set_label(colormap_label, size=20)
        
    if colorbarway == True:
    
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('bottom', size='3%', pad=0.03)
        
        norm=plt.Normalize(vmin = np.min(node_colors), vmax=np.max(node_colors))
        
        fig1.colorbar( mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, extend='max', orientation='horizontal')
        cax.set_xlabel(colormap_label, size=20)
        
    
    if make_new_figure : plt.show()

    return ;