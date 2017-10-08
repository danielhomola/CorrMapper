import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#------------------------------------------------------------------------------
#
#                C O L L A T I N G  R E S U L T  F I L E S 
#
#------------------------------------------------------------------------------


# input folder is either results or results_topvar
input_folder = "/home/daniel/corr/supp/cmTest/results2"
os.chdir(input_folder)
# output folder is where you want to save all figures and tables
output_folder = "/home/daniel/corr/supp/cmTest/figures_tables"

fs = os.listdir('.')

# open a result file to get the methods, etc
p = pd.read_csv(fs[0])
# define the columns of the final results DataFrame
col_names = p['Method'].values

# helper function which takes in a file name and extracts the sample, feature,
# informative feature numbers and the random state
def get_exp_info(filename):
    pat =  r"samp_(\d+)_feat_(\d+)_rand_(\d+)"
    sample = int(re.match(pat, filename).group(1))
    feature = int(re.match(pat, filename).group(2))
    random_seed = int(re.match(pat, filename).group(3))
    return [sample, feature, random_seed]

# define the multi-level index, extract sample and feature nums first
ind = [get_exp_info(x) for x in fs]
s_f_ratio = [x[0]/float(x[1]) for x in ind]
for i,x in enumerate(ind):
    x.append(s_f_ratio[i])
ind = [tuple(x) for x in ind]
ind = pd.MultiIndex.from_tuples(ind, names=['Samples','Features', 'RandomSeed',
                                            'Ratio'])

# DataFrame for sens | prec merged results, for the report
results = pd.DataFrame(columns=col_names, index=ind)

# DataFrames for sens and prec values separately for analysis
sens = pd.DataFrame(columns=col_names, index=ind)
prec = pd.DataFrame(columns=col_names, index=ind)
# reorder the multi level index to make it easier to read
results = results.sort_index()
sens = sens.sort_index()
prec = prec.sort_index()

# iterate all result files
for i, f in enumerate(fs):    
    t = pd.read_csv(f ,index_col=0)
    
    # get indexing for multi level index in the results DF
    row = get_exp_info(f)
    row.append(row[0]/float(row[1]))
    row = tuple(row)
    # create the values of the cells
    recall = t[' Recall']
    precision = t[' Prec']
    merged_val = zip(recall, precision )
    merged_val = ["%s | %s" % ('{0:.4f}'.format(x[0]), '{0:.4f}'.format(x[1])) for x in merged_val]
        
    # write to DataFrames
    results.loc[row, t.index.values] = merged_val
    sens.loc[row, t.index.values] = recall
    prec.loc[row, t.index.values] = precision


#------------------------------------------------------------------------------
#
#                 A N A L Y S I S   O F   R E S U L T S
#
#------------------------------------------------------------------------------

def save_table_to_latex(df, name, floats=False):
    filename = os.path.join(output_folder, name + '.tex')
    latex_f = open(filename, 'w')
    if not floats:
        latex_f.write(df.to_latex(escape=False))
    else:
        latex_f.write(df.to_latex(escape=False, float_format=lambda x: '{0:.4f}'.format(x)))
    latex_f.close()

# rename columns to pretty names
method_names = ['UnivarFDR','L1 SVC', 'Boruta', 'JMI', 'GraphLasso', 
'Marginal 0.05', 'Marginal 0.1', 'Marginal 0.2', 'Marginal 0.3', 
'Marginal 0.5', 'Marginal 0.7', 'Marginal 0.8', 'mixOmics 0.05',
'mixOmics 0.1', 'mixOmics 0.2', 'mixOmics 0.3', 'mixOmics 0.5', 
'mixOmics 0.7', 'mixOmics 0.8' ]
sens.columns = method_names
prec.columns = method_names

# convert to float so we can do groubpy and otherstuff
sens = sens.astype(float)
prec = prec.astype(float)

#------------------------------------------------------------------------------
# CREATE SUMMARY TABLE OF METHODS

# prepare summary table
def get_merged_vals(col1, col2):
    merged = zip(col1, col2)
    merged = ["%s $\pm$ %s" % ('{0:.4f}'.format(x[0]), '{0:.2f}'.format(x[1])) for x in merged]
    return pd.Series(merged, index=col1.index)
    
def concat_summary_table(sens, prec):
    s_merged = get_merged_vals(sens.mean(), sens.std())
    p_merged = get_merged_vals(prec.mean(), prec.std())
    sp_merged = get_merged_vals(prec.mean() + sens.mean(), prec.std()+sens.std())
    summary = pd.concat([prec.count(), s_merged, p_merged, sp_merged], axis=1)
    
    summary.columns = ["Count", "Rec", "Prec", "R + P"]
    return summary

summary = concat_summary_table(sens, prec)
save_table_to_latex(summary, "cm_mixomics_summary", True)
