import os
import pandas as pd
import numpy as np
from backend.utils.check_uploaded_files import open_file


def write_tables(params, name, annot_file, r, p, row2bin, bin2label, axis=0):
    """
    Filters, extends and saves annot_file as .json for DataTables JS library +
    writes 2nd half of _var.js. axis=1 corresponds to rows, axis=0 to columns
    """

    # --------------------------------------------------------------------------
    # FILTER, EXTEND ANNOT FILE
    # --------------------------------------------------------------------------

    # filter annotation file and R matrix by the GEs used in the network
    if axis == 1:
        annot = annot_file.loc[r.index]
    else:
        annot = annot_file.loc[r.columns]

    # drop Bin columns we don't need it anymore
    annot.drop('Bin', axis=1, inplace=True)

    # add number of correlated features column to annot file
    corr_num = np.sum(r != 0, axis=axis)
    annot.insert(3, '#corr', corr_num)

    # loop through annot files and add the as separate columns
    #   - name of correlated features,
    #   - r-val of correlated features,
    #   - p-val of correlated features,
    #   - name of bin in network

    corr_name = []
    r_vals = []
    p_vals = []
    corr_bin = []
    if axis == 0:
        r = r.T
        p = p.T

    for row in r.index:
        # names of correlated features
        non_zero = r.loc[row][r.loc[row] != 0]
        non_zero_names = non_zero.index.values.ravel()
        # r-vals of correlated features
        non_zero_r_vals = np.array(map("{0:.3f}".format, non_zero.values.ravel()))
        # p-vals of correlated features
        non_zero_p = p.loc[row][r.loc[row] != 0]
        non_zero_p_vals = np.array(map("{0:.7f}".format, non_zero_p.values.ravel()))

        # sort correlated features alphabetically
        if len(non_zero_names) > 1:
            ind_sort = np.argsort(non_zero_names)
            non_zero_names = non_zero_names[ind_sort]
            non_zero_r_vals = non_zero_r_vals[ind_sort]
            non_zero_p_vals = non_zero_p_vals[ind_sort]

        # merge values intro one string
        corr_name.append('__|__'.join(non_zero_names))
        r_vals.append('__|__'.join(non_zero_r_vals))
        p_vals.append('__|__'.join(non_zero_p_vals))

        # bin name of row
        corr_bin.append(bin2label[row2bin[row]])

    annot.insert(0, 'Name', annot.index.values.ravel())
    annot.insert(0, 'CorrMapperPval', p_vals)
    annot.insert(0, 'CorrMapperRval', r_vals)
    annot.insert(0, 'CorrMapperCorrName', corr_name)
    annot.insert(0, 'CorrMapperBin', corr_bin)

    # --------------------------------------------------------------------------
    # WRITE MODIFIED ANNOT FILE AS JSON
    # --------------------------------------------------------------------------

    json_str = annot.to_json(None, orient='records')
    # make it legible by breaking line and adding tabulation
    n1 = '    '
    n2 = n1 * 2
    json_str = json_str.replace('","','",\n' + n2 + '"')
    json_str = json_str.replace('},{','},\n' + n2 + '{')
    # add prefix and suffix so Data-tables accepts it
    json_str = '{\n' + n1 + '"data":\n' + n2 + json_str + '\n}'

    # save results
    if axis == 1:
        table = '_row-table.json'
        table_var = 'var table1_cols'
    else:
        table = '_column-table.json'
        table_var = 'var table2_cols'

    file_name = file_name = name + table
    out_file = os.path.join(params['vis_genomic_folder'], file_name)
    f = open(out_file, 'w')
    f.write(json_str)
    f.close()

    # --------------------------------------------------------------------------
    # WRITE SECOND HALF OF VARS.JS
    # --------------------------------------------------------------------------

    file_name = name + '_vars.js'
    out_file = os.path.join(params['vis_genomic_folder'], file_name)
    f = open(out_file, 'a')
    f.write(table_var + ' = ' + str(list(annot.columns[4:])) + ';\n')
    f.close()
