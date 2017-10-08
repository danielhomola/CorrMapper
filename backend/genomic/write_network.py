import os
import pandas as pd
import numpy as np
from backend.utils.check_uploaded_files import open_file


def write_network(params, name, annot, r, sym=False):
    """
    Writes _network.js and first half of _var.js
    """
    
    # --------------------------------------------------------------------------
    # OPEN BIN AND CHROMO FILE, BUILD DICTS FROM THEM
    # --------------------------------------------------------------------------
    bin_file = params['bin_file']
    chromo_file = ''.join(bin_file.split('__')[:-1]) + '__chromosomes.txt'
    bins = pd.read_table(bin_file)
    chromo = pd.read_table(chromo_file)

    bin2label = {}
    label2chromosome = {}
    chr_starts = {}
    # list holding labels in their biological order
    labels = []
    for i in bins.index:
        row = bins.loc[i]
        # calculate 10 megabase units
        start ="{0:.1f}".format(row.Start / 10000000.)
        end = "{0:.1f}".format(row.End / 10000000.)
        label = 'Chr' + str(row.Chromosome) + '_' + start + '-' + end
        bin2label[row.Bin] = label
        label2chromosome[label] = "chr_" + str(row.Chromosome)
        if row.Chromosome not in chr_starts:
            chr_starts[row.Chromosome] = int(row.ChromoStart)
        labels.append(label)
    
    # --------------------------------------------------------------------------
    # DEFINE DICTS AND LISTS FOR NETWORK
    # --------------------------------------------------------------------------

    # filter annot files to R matrix
    annot1 = annot[0].loc[r.index]
    annot2 = annot[1].loc[r.columns]

    # build dicts that maps feature names to bin numbers + saves used chroms
    row2bin = {}
    for i in annot1.index:
        row = annot1.loc[i]
        row2bin[i] = row.Bin

    col2bin = {}
    for i in annot2.index:
        row = annot2.loc[i]
        col2bin[i] = row.Bin

    # --------------------------------------------------------------------------
    # BUILD DICT OF EDGES, CALC EDGE AND NODE METRICS
    # --------------------------------------------------------------------------

    edges = {}
    used_chromosomes = {}
    # build an edge dictionary, also if needed discard overlapping GEs
    for ri, row in enumerate(r.index):
        for ci, col in enumerate(r.columns):
            # if sym, only use the upper triangle of the r matrix
            if not sym or (sym and ci > ri):
                cell = r[col].loc[row]
                if cell != 0:
                    # check if GEs overlap, and if their chromos are new
                    source = bin2label[row2bin[row]]
                    target = bin2label[col2bin[col]]

                    # check the chromosomes, add them if they are new
                    if label2chromosome[source] not in used_chromosomes:
                        used_chromosomes[label2chromosome[source]] = 1
                    if label2chromosome[target] not in used_chromosomes:
                        used_chromosomes[label2chromosome[target]] = 1

                    # add edge tp network
                    edge = source + '__|__' + target
                    if edge not in edges:
                        edges[edge] = []
                    edges[edge].append(cell)

    # calculate
    #  - node size: the total number of correlations in a bin
    #  - edge size: dict with the number of correlations between two bins
    #  - edge_corr: the median correlation for an edge between two bins
    node_size = {}
    edge_size = {}
    edge_corr = {}
    for edge, r_vals in edges.iteritems():
        # save size of edges and median corr for colouring
        size = len(r_vals)
        edge_size[edge] = size
        edge_corr[edge] = np.median(np.array(r_vals))

        # increase node sizes using two ends of the edge
        n1, n2 = edge.split('__|__')
        if n1 not in node_size:
            node_size[n1] = 0
        if n2 not in node_size:
            node_size[n2] = 0
        node_size[n1] += size
        node_size[n2] += size

    # save max edge and node size and normalize the edge and node size
    max_node_size = np.max(node_size.values())
    # max node size is 20 so radius must be between 2 (so it's visible) and 10
    if max_node_size < 5:
        node_size = {k: 2 + v/float(max_node_size) * 4 for k,v in node_size.iteritems()}
    else:
        node_size = {k: 2 + v/float(max_node_size) * 8 for k,v in node_size.iteritems()}
    # max edge width is 5 in HTML
    max_edge_size = np.max(edge_size.values())
    if max_edge_size < 4:
        edge_size_normalized = {k: 1 + v / float(max_edge_size) * 2 for k,v in edge_size.iteritems()}
    else:
        edge_size_normalized = {k: 1 + v / float(max_edge_size) * 4 for k,v in edge_size.iteritems()}

    # --------------------------------------------------------------------------
    # WRITE NETWORK
    # --------------------------------------------------------------------------
    file_name = name + '_network.js'
    out_file = os.path.join(params['vis_genomic_folder'], file_name)
    f = open(out_file, 'w')

    # write parent nodes, i.e. chromosomes of nodes
    f.write('var parentsData = [\n')
    out = ''

    # build list of parent nodes (chromosomes) for hierarchical edge bundling
    chrs = []
    for c in chromo.Chromosome.values:
        chr = 'chr_' + str(c)
        if chr in used_chromosomes:
            chrs.append(chr)
    chr2num = {c:ci for ci, c in enumerate(chrs)}

    # write the parent nodes
    for c in chrs:
        out += ('{"name":"%s", "chr_num":%d},\n'
                % (c, chr2num[c]))
    # clip last coma, otherwise we get js error
    out = out[:-2]
    out += '\n];\n\n'
    f.write(out)

    # write nodes
    f.write('var nodesData = [\n')
    out = ''
    for label in labels:
        if label in node_size:
            out += ('{"name":"%s", "chromosome":"%s", "chr_num":%d,'
                    ' "nodeSize":%f},\n'
                    % (label, label2chromosome[label],
                       chr2num[label2chromosome[label]], node_size[label]))
    out = out[:-2]
    out += '\n];\n\n'
    f.write(out)

    # write edges
    f.write('var edgesData = [\n')
    out = ''
    for edge in edges:
        source, target = edge.split('__|__')
        out += ('{"source":"%s", "target":"%s", "edgeCorr":%f, '
                '"edgeWidth": %f, "edgeNum":%d},\n'
                % (source,
                   target,
                   edge_corr[edge],
                   edge_size_normalized[edge],
                   edge_size[edge]))
    out = out[:-2]
    out += '\n];\n\n'
    f.write(out)
    f.close()

    # --------------------------------------------------------------------------
    # WRITE FIRST HALF OF VARS.JS
    # --------------------------------------------------------------------------

    file_name = name + '_vars.js'
    out_file = os.path.join(params['vis_genomic_folder'], file_name)
    f = open(out_file, 'w')

    r_min = np.min(r.values)
    r_max = np.max(r.values)
    # if sym only use the lower triangle without the diagonal
    if sym:
        r_min = np.min(r.values[np.tril_indices_from(r.values, k=-1)])
        r_max = np.max(r.values[np.tril_indices_from(r.values, k=-1)])

    f.write('var maxLink = ' + str(max_edge_size) + ';\n')
    f.write('var minCorr = ' + str(r_min) + ';\n')
    f.write('var maxCorr = ' + str(r_max) + ';\n')
    f.close()

    return bin2label, row2bin, col2bin
