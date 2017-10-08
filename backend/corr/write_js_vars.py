"""
Utility functions for D3 correlation maps.

Functions to:
 - write matrix of heatmap into javascript readable .csv
 - write many variables for the given heatmap / visualisation
 - write the variables for the given network visualisations
"""

import pandas as pd
import numpy as np
import matplotlib
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from colour import Color

# -----------------------------------------------------------------------------
#                       WRITE HEATMAP MATRIX FOR JS
# -----------------------------------------------------------------------------

def matrix_for_d3(r, p, sym, out_file):
    """
    Writes a csv file from the heatmap matrix in JS friendly format.
    """
    n, m = r.shape
    f = open(out_file, 'w')
    f.write('rowIdx,colIdx,rVal,pVal\n')
    for ri in range(n):
        for ci in range(m):
            # if sym, only show the upper triangle of the heatmap
            # if not sym or (sym and ci >= ri):
            rval = r.iloc[ri,ci]
            pval = p.iloc[ri,ci]
            if rval == 0. or (sym and ri == ci):
                rval = '0'
                pval = 'nan'
            else:
                rval = "{0:.3f}".format(rval)
                pval = "{0:.7f}".format(pval)
            row = str(ri)
            col = str(ci)
            f.write(','.join([row, col, rval, pval])+'\n')
    f.close()
    
# -----------------------------------------------------------------------------
#                           WRITE VARIABLES FOR JS
# -----------------------------------------------------------------------------

def str_list2js(l):
    result = '[\'' + '\',\''.join(map(str,l)) + '\']'
    return result


def get_reordered_fold_change(axis_labels, fold_change):
    fc = []
    for i in axis_labels:
        if i in fold_change:
            fc.append("{0:.2f}".format(fold_change[i]))
        else:           
            fc.append('')
    return fc


def write_vars_for_d3(data, names, out_file, fold_change, sym, modules=list(),
                      data_clusters_col=list(), data_clusters_row=list(),
                      r_clusters_col=list(), r_clusters_row=list(),
                      name_order_col=list(), name_order_row=list()):
    """
    Writes the various variables for vis.js.
    """
    n, m = data.shape
    f = open(out_file, 'w')
    
    row_short_names = [names[i]['short_name'] for i in data.index]
    row_long_names = [names[i]['long_name'] for i in data.index]    
    col_short_names = [names[i]['short_name'] for i in data.columns]
    col_long_names = [names[i]['long_name'] for i in data.columns]
    longest_row = np.max([len(i) for i in row_long_names])
    longest_col = np.max([len(i) for i in col_short_names])
    row_fold_change = get_reordered_fold_change(row_long_names, fold_change)
    col_fold_change = get_reordered_fold_change(col_long_names, fold_change)

    # if sym:
    #     # ignore diagonal ones if the matrix is symmetric
    #     max_data = np.max(data.values[np.tril_indices(n, k=-1)])
    #     min_data = np.min(data.values[np.tril_indices(n, k=-1)])
    # else:
    max_data = np.max(data.values)
    min_data = np.min(data.values)

    # write basic vars for vis
    f.write('var rowNumber = ' + str(n) + ';\n')
    f.write('var colNumber = ' + str(m) + ';\n')
    f.write('var maxData = ' + str(max_data) + ';\n')
    f.write('var minData = ' + str(min_data) + ';\n')
    f.write('var longestRow = ' + str(longest_row) + ';\n')
    f.write('var longestCol = ' + str(longest_col) + ';\n')

    # write labels of rows and columns
    f.write('var rowLabel = ' + str_list2js(data.index.values) + ';\n')
    f.write('var colLabel = ' + str_list2js(data.columns.values) + ';\n')
    f.write('var rowLabelShort = ' + str_list2js(row_short_names) + ';\n')    
    f.write('var rowLabelLong = ' + str_list2js(row_long_names) + ';\n')
    f.write('var colLabelShort = ' + str_list2js(col_short_names) + ';\n')    
    f.write('var colLabelLong = ' + str_list2js(col_long_names) + ';\n')

    # write out modules of default ordering (mods is a list of strings)
    f.write('var modules = ' + str_list2js(modules) + ';\n')

    # write orderings
    f.write('var defaultRowOr = ' + str(range(n)) + ';\n')
    f.write('var defaultColOr = ' + str(range(m)) + ';\n')
    f.write('var dataRowOr = ' + str(data_clusters_row) + ';\n')
    f.write('var dataColOr = ' + str(data_clusters_col) + ';\n')
    f.write('var rRowOr = ' + str(r_clusters_row) + ';\n')
    f.write('var rColOr = ' + str(r_clusters_col) + ';\n')
    f.write('var nameRowOr = ' + str(list(name_order_row)) + ';\n')
    f.write('var nameColOr = ' + str(list(name_order_col)) + ';\n')

    # write fold change vars    
    f.write('var rowFoldChange = ' + str(row_fold_change) + ';\n')
    f.write('var colFoldChange = ' + str(col_fold_change) + ';\n')
    f.close()

# -----------------------------------------------------------------------------
#                    WRITE VARIABLES FOR NETWORK IN JS
# -----------------------------------------------------------------------------

def is_last(d, i):
    if i == len(d.keys())-1:
        last = True
    else:
        last = False
    return last


def write_dict(ks, vs, string_value, is_last):
    s = '{'
    for i, k in enumerate(ks):
        if i != 0:
            s += ','
        # first add the key which must be a string
        s += '"'+ str(k) +'":'
        # then the value
        if string_value[i]:
            s += '"' + str(vs[i]) + '"'
        else:
            s += str(vs[i])
    # close the last child
    if is_last:
        s += '}\n'
    else:
        s += '},\n'
    return s


def get_color(gradient, r_val, my_norm):
    g = int(my_norm(r_val)*100)
    if g >= 100:
        g = 99
    elif g < 0:
        g = 0
    return gradient[g]


def get_width(r_val, r):
    r_min = np.abs(r.values).min()
    r_val = np.abs(r_val) - r_min
    r_max = np.abs(r.values).max()
    r_val /= float(r_max)
    r_val *= 2
    return r_val


def get_rvals(k, axis, r, names):
    rvals = ''
    if axis == 1:
        row = r.loc[k]
        row = row[row != 0]
        for i in row.index:
            if i != k:
                rvals += names[i]['short_name'] + ':\t'
                rvals += "{0:.2f}".format(row.loc[i]) + '<br>'
    else:
        col = r[k]
        col = col[col != 0]
        for i in col.index:
            if i != k:
                rvals += names[i]['short_name'] + ':\t'
                rvals += "{0:.2f}".format(col.loc[i]) + '<br>'
    return rvals


def write_network_for_d3(B, r, names, fold_change, out_file):
    """
    Write the variables for the network visualisation.
    """
    f = open(out_file, 'w')
    
    # d3plus cannot draw circles and squares in the same network yet so we use
    # two colors to draw bipartite graphs
    top_color = '#fff'
    bottom_color = '#333'

    # found a better way in JS to handle this
    #col_min = "#0971B2"
    #col_max = "#CE1212"
    #grad1 = [c.hex for c in Color(col_min).range_to(Color("white"), 50)]
    #grad2 = [c.hex for c in Color("white").range_to(col_max, 51)]
    #gradient = grad1 + grad2[1:]
    #my_norm = matplotlib.colors.Normalize(r.values.min(), r.values.max(), True)

    # get params of the network
    top_nodes = set(n for n,d in B.nodes(data=True) if d['bipartite'] == 0)
    pos = graphviz_layout(B)
    pos_mins = pd.DataFrame(pos.values()).min().values
    x_min = pos_mins[0]
    y_min = pos_mins[1]

    # NETWORK DATA
    f.write('var network_data = [\n')
    for i, k in enumerate(pos):
        # is it a row or col label        
        if k in r.index:
            axis = 1
        else:
            axis = 0
            
        # is it a top-node in a bipartite graph
        if k in top_nodes:
            color = top_color
        else:
            color = bottom_color
            
        # names of the nodes, short and long
        short_label = names[k]['short_name']
        long_label = names[k]['long_name']
        
        # fold change
        if long_label in fold_change:
            fc = "{0:.2f}".format(fold_change[long_label])
        else:
            fc = ''
        
        rvals = get_rvals(k, axis, r, names)        
        ks = ['name', 'shortLabel', 'longLabel', 'color', 'foldChange', 
              'R<sup>2<sup>']
        vs = [k, short_label, long_label, color, fc, rvals]
        f.write(write_dict(ks, vs, [True] * 6, is_last(pos, i)))
    f.write('];\n')

    # POSITIONS
    f.write('var positions = [\n')
    for i, k in enumerate(pos):
        ks = ['name', 'x', 'y']
        x = (pos[k][0] - x_min)
        y = (pos[k][1] - y_min)
        vs = [k, x, y]
        f.write(write_dict(ks, vs, [True, False, False], is_last(pos, i)))
    f.write('];\n')

    # CONNECTIONS
    f.write('var connections = [\n')
    for i, e in enumerate(B.edges_iter()):
        if i == len(B.edges()) - 1:
            last = True
        else:
            last = False

        if e[0] != e[1]:
            r_val = B.get_edge_data(e[0],e[1])['r']
            p_val = B.get_edge_data(e[0],e[1])['p']

            # found a better way in JS to handle this
            # color = get_color(gradient, r_val, my_norm)

            width = get_width(r_val, r)
            r_val = "{0:.5f}".format(r_val)
            p_val = "{0:.5f}".format(p_val)

            ks = ['source', 'target', 'width', 'rval', 'pval']
            vs = [e[0], e[1], width, r_val, p_val]
            f.write(write_dict(ks, vs, [True, True, False, False, False], last))
    f.write('];\n')
    f.close()
