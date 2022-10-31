#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:23:15 2022

@author: shihab
"""
import csv
#from copy import copy
import os
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np # array manipulation
#import pandas as pd # data manipulation
#import seaborn as sns # plotting
#import dc_stat_think as dcst
#import statistics
#import itertools
#from matplotlib.lines import Line2D

#from numpy.linalg import svd
#from sklearn.decomposition import TruncatedSVD
#from sklearn.manifold import TSNE
#from scipy import spatial
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn import manifold
#from sklearn.metrics import pairwise_distances
#import datetime
#import hdbscan
#import pickle
#import sys
#import unittest

G = nx.DiGraph()

def sumforline(FILEPATH):
    with open(FILEPATH, encoding="utf8") as f:
        return sum(1 for line in f)

def readcsv(FILEPATH, event):
    epoch_time = []
    print("Experiment for first: ", event)
    with open(FILEPATH, 'r', encoding="utf8") as file:
        reader = csv.reader(file)
        i=0
        for row in reader:
            if (i>event):
                break
            
            Time = row[0]
            Duration = row[1]
            SrcDevice = row[2]
            DestDevice = row[3]
            
            
            Concat_Time = []
            epoch_time.append(Time)
    
            if not G.has_edge(SrcDevice, DestDevice):
                Concat_Time.append(Time)
                G.add_edge(SrcDevice, DestDevice, Arrival = Concat_Time, Dur = Duration)
            else:
                Concat_Time = G[SrcDevice][DestDevice]['Arrival']
                Concat_Time.append(Time)
                G[SrcDevice][DestDevice]['Arrival'] = Concat_Time
                
            
            i+=1
        print("Number of Nodes: ", len(G.nodes))
        print("Number of Edges: ", len(G.edges))
        print("Start End Diff", epoch_time[0], " ", epoch_time[len(epoch_time)-1])
        total_time = int(epoch_time[len(epoch_time)-1]) - int(epoch_time[0])
        return total_time
    
def calculate_node_class(node_num):
    
    print("Number of Nodes: ", node_num)
    indegree_list = G.in_degree(G.nodes)
    outdegree_list = G.out_degree(G.nodes)
    
    indegree = 0
    outdegree = 0
    totaldegree = 0
    category = 'client'
    
    peer = 0
    client = 0
    server = 0
    
    nx.set_node_attributes(G, indegree, 'indegree')
    nx.set_node_attributes(G, outdegree, 'outdegree')
    nx.set_node_attributes(G, totaldegree, 'total_degree')
    nx.set_node_attributes(G, category, 'category')
      
    for ((n1, ind),(n2, outd)) in zip(indegree_list, outdegree_list):

        G.nodes[n1]['indegree'] = ind
        G.nodes[n2]['outdegree'] = outd
        G.nodes[n1]['total_degree'] = ind + outd
        if ind > 0 and outd >0:
            G.nodes[n1]['category'] = 'peer'
            peer+=1
        elif ind == 0:
            G.nodes[n1]['category'] = 'client'
            client+=1
        else:
            G.nodes[n1]['category'] = 'server'
            server+=1
            
    print("Number of Client: ", client)
    print("Number of Server: ", server)
    print("Number of Peer: ", peer)
    return client+server+peer

    
def draw_connected_component():
    
    
    node_number_list = []
    edge_number_list = []
    for c in sorted(nx.weakly_connected_components(G),key=len, reverse=True):
        node_number_list.append(len(c))
        s = G.subgraph(c)
        edge_number_list.append(len(s.edges))
        
    #print(node_number_list)
    #print(edge_number_list)
    
    x = np.arange(len(node_number_list))
    width = .40
    
    fig, ax = plt.subplots(figsize=(50,22))
    ax.bar(x - width/2, node_number_list, width, label='Number of nodes', color='blue', log=True)
    ax.bar(x + width/2, edge_number_list, width, label='Number of edges', color='orange', log=True)
    plt.legend(loc="upper right", prop={'size': 25})
    plt.xlabel('Number of Connected Componets', fontdict=None, loc='center', fontsize = 25)
    plt.ylabel('Number of Nodes and Edges', fontdict=None, loc='center', fontsize = 25)
    #plt.bar_label(bar1, rotation=90, fontsize=20, padding= 10)
    #plt.bar_label(bar2, rotation=90, fontsize=20, padding= 10)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    fig.savefig("Connected_Componets_day_" + str(events) + ".jpeg")   

    
def find_giant_component():
    GC =sorted(nx.weakly_connected_components(G),key=len, reverse=True)
    giant_component = G.subgraph(GC[0])
    print("Number of Components: ", len(GC))
    print("Number of Nodes in Giant Component: ", len(giant_component.nodes()))
    print("Number of Edges in Giant Component: ", len(giant_component.edges()))
    save_graph(giant_component, processed_data_path, FILE_NAME)
    return len(GC)


def save_graph(gr, savepath, file):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    nx.write_gml(gr, savepath + file)
    print(savepath + file) 
 
    
print("---------Read_CSV----------------")
#############Assign Path Vaiables Here#####################

PATH = 'sample_1.csv'

total_row = sumforline(PATH)
print("Number of rows in CSV: ", total_row)

events = total_row



processed_data_path = "../Processed_Data/"
FILE_NAME = "giant_component_" + str(events) + ".gml"
########################################################## 


e_time = readcsv(PATH, events)
total_node = calculate_node_class(len(G.nodes))
draw_connected_component()
g_len = find_giant_component()

 
"""  
class unit_test(unittest.TestCase):
      
    def test_sumforline(self):
        self.assertEquals(sumforline("sample_1.csv"), 11)
        self.assertEquals(sumforline("split_0.csv"), 1000001)
        self.assertEquals(sumforline("split_1.csv"), 10001)
        
    def test_readcsv(self):
        self.assertEquals(readcsv("sample_1.csv", 11), 39)
        self.assertEquals(readcsv("split_0.csv", 100), 244)
        self.assertEquals(readcsv("split_1.csv", 100), 7)
        
    def test_calculate_node_class(self):
        self.assertEquals(calculate_node_class(G), len(G.nodes))
        
    def test_find_giant_component(self):
        self.assertGreaterEqual(len(G.nodes), calculate_node_class(G))
        
      
if __name__ == '__main__':
   unittest.main()
"""