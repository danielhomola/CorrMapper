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
input_folder = "/home/daniel/corr/supp/cmTest/results"
os.chdir(input_folder)
# output folder is where you want to save all figures and tables
output_folder = "/home/daniel/corr/supp/cmTest/figures_tables"

fs = os.listdir('.')

# open a result file to get the methods, etc
p = pd.read_csv(fs[0])
# define the columns of the final results DataFrame
col_names = p['FS Method'].values

# helper function which takes in a file name and extracts the sample, feature,
# informative feature numbers and the random state
def get_exp_info(filename):
    pat =  r"samp_(\d+)_feat_(\d+)_inf_(\d+)_star_(\d+)_rand_(\d+)"
    sample = int(re.match(pat, filename).group(1))
    feature = int(re.match(pat, filename).group(2))
    informative = int(re.match(pat, filename).group(3))
    star = int(re.match(pat, filename).group(4))/100.
    random_seed = int(re.match(pat, filename).group(5))
    return [sample, feature, informative, star, random_seed]

# define the multi-level index, extract sample and feature nums first
ind = [get_exp_info(x) for x in fs]
s_f_ratio = [x[0]/float(x[1]) for x in ind]
for i,x in enumerate(ind):
    x.append(s_f_ratio[i])
ind = [tuple(x) for x in ind]
ind = pd.MultiIndex.from_tuples(ind, names=['Samples','Features','Informative',
                                            'Star', 'RandomSeed', 'Ratio'])

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
    # filter out random last rows
    t = t.loc[t.index[[i in ["fdr", "l1svc", "boruta", "jmi"] for i in t.index]]]
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
method_names = ['UnivarFDR','L1 SVC', 'Boruta', 'JMI']
sens.columns = method_names
prec.columns = method_names

# convert to float so we can do groubpy and otherstuff
sens = sens.astype(float)
prec = prec.astype(float)

# separate tables by star then drop it as column
sens = sens.reset_index(level="Star")
prec = prec.reset_index(level="Star")

sens05 = sens[sens.Star == 0.05][method_names]
# mistyped stars, so 0.01 is actually 0.1
sens1 = sens[sens.Star == 0.01][method_names]
prec05 = prec[prec.Star == 0.05][method_names]
prec1 = prec[prec.Star == 0.01][method_names]

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
    summary = pd.concat([s_merged, p_merged, sp_merged], axis=1)
    
    summary.columns = ["Rec", "Prec", "R + P"]
    return summary


# create table for ratios above .2 and 0, i.e. all results
ratios = [0, 0.2]
for ratio in ratios:
    sens05_tmp = sens05.reset_index("Ratio")
    sens05_tmp = sens05_tmp[sens05_tmp.Ratio >= ratio]
    sens05_tmp.drop("Ratio", axis=1 , inplace=True)
    sens1_tmp = sens1.reset_index("Ratio")
    sens1_tmp = sens1_tmp[sens1_tmp.Ratio >= ratio]
    sens1_tmp.drop("Ratio", axis=1 , inplace=True)    
    prec05_tmp = prec05.reset_index("Ratio")
    prec05_tmp = prec05_tmp[prec05_tmp.Ratio >= ratio]
    prec05_tmp.drop("Ratio", axis=1 , inplace=True)
    prec1_tmp = prec1.reset_index("Ratio")
    prec1_tmp = prec1_tmp[prec1_tmp.Ratio >= ratio]
    prec1_tmp .drop("Ratio", axis=1 , inplace=True)
    
    summary05 = concat_summary_table(sens05_tmp, prec05_tmp)
    summary1 = concat_summary_table(sens1_tmp, prec1_tmp)
    summary = pd.concat([summary05.T, summary1.T])
    summary.insert(0, "StARS", [0.05]*3+[0.1]*3)
    filename = "summary_%f" % ratio
    save_table_to_latex(summary, filename, True)
