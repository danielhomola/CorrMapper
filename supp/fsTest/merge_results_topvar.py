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
input_folder = "/home/daniel/corr/supp/fsTest/results_topvar"
os.chdir(input_folder)
# output folder is where you want to save all figures and tables
output_folder = "/home/daniel/corr/supp/fsTest/benchmark2_figures_tables"

fs = os.listdir('.')

# open a result file to get the methods, etc
p = pd.read_table(fs[0])
# define the columns of the final results DataFrame
col_names = p.Method.values

# helper function which takes in a file name and extracts the sample, feature,
# informative feature numbers and the random state
def get_exp_info(filename):
    pat =  r"samp_(\d+)_feat_(\d+)_inf_(\d+)_rel_(\d+)_rand_(\d+)"
    sample = int(re.match(pat, filename).group(1))
    feature = int(re.match(pat, filename).group(2))
    relevant = int(re.match(pat, filename).group(4))
    random_seed = int(re.match(pat, filename).group(5))
    informative = int(re.match(pat, filename).group(3))/float(feature)
    return [sample, feature, relevant, random_seed, informative]

# define the multi-level index, extract sample and feature nums first
ind = [get_exp_info(x) for x in fs]
s_f_ratio = [x[0]/float(x[1]) for x in ind]
for i,x in enumerate(ind):
    # NOTE, the ratio is FIXED to 0.5 by run_benchmark_topvar
    x.append(0.5)
ind = [tuple(x) for x in ind]
ind = pd.MultiIndex.from_tuples(ind, names=['Samples','Features','Relevant',
                                            'RandomSeed', 'Informative', 'Ratio'])

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
    try:
        t = pd.read_table(f ,index_col=0)
        # get indexing for multi level index in the results DF
        row = get_exp_info(f)
        # NOTE, the ratio is FIXED to 0.5 by run_benchmark_topvar
        row.append(0.5)
        row = tuple(row)
        # create the values of the cells
        merged_val = zip(t.Sens, t.Prec)
        merged_val = ["%s | %s" % ('{0:.4f}'.format(x[0]), '{0:.4f}'.format(x[1])) for x in merged_val]
            
        # write to DataFrames
        results.loc[row, t.index.values] = merged_val
        sens.loc[row, t.index.values] = t.Sens
        prec.loc[row, t.index.values] = t.Prec
    except:
        pass

#------------------------------------------------------------------------------
#
#                 A N A L Y S I S   O F   R E S U L T S
#
#------------------------------------------------------------------------------

# convert to float so we can do groubpy and otherstuff
sens = sens.astype(float)
prec = prec.astype(float)

# rename columns to pretty names
method_names = ['UnivarPerc', 'UnivarFDR', 'RFE CV', 'L1 SVC', 
                'StabSel', 'Boruta', 'JMI']
sens.columns = method_names
prec.columns = method_names

#------------------------------------------------------------------------------
# CREATE TABLES FOR PLOTTING

#------------------------------------------------------------------------------
# calculate means and stds by Ratio groups for sensitivity
smean = sens.groupby(level='Informative').mean()
sstd = sens.groupby(level='Informative').std()

# calculate means and stds by Ratio groups for precision
pmean = prec.groupby(level='Informative').mean()
pstd = prec.groupby(level='Informative').std()

# generate multi level index for columns for sensitivity
spmean = pd.concat([smean, pmean], axis=1)
sens_prec_names = ['Recall']*len(col_names) + ['Precision']*len(col_names)
multi_cols = zip((sens_prec_names), spmean.columns)
multi_cols = pd.MultiIndex.from_tuples(multi_cols, names=['','Method'])
spmean.columns = multi_cols

# generate multi level index for columns for precision
spstd = pd.concat([sstd, pstd], axis=1)
multi_cols = zip((sens_prec_names), spstd.columns)
multi_cols = pd.MultiIndex.from_tuples(multi_cols, names=['SP','Method'])
spstd.columns = multi_cols

#------------------------------------------------------------------------------
#
#                 P L O T T I N G   O F   R E S U L T S
#
#------------------------------------------------------------------------------

# setup seaborn style
colors = ["windows blue", "faded green"]
sns.set_palette(sns.xkcd_palette(colors))
sns.set_style("whitegrid", {"grid.color": ".9"})
sns.set_context("talk")

nrows = 1
ncols = 2
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
fig.set_figheight(3.5)
fig.set_figwidth(8.27)
i = 0
ratios_to_plot = range(2)
for c in range(ncols):
    ratio = ratios_to_plot[c]
    means = spmean.iloc[ratio,].T.unstack().T
    means.index.name=""
    error_bars = spstd.iloc[ratio,].T.unstack().T
    means.plot(ax=ax[c], kind='barh', stacked=False, 
               xerr=error_bars,xlim=(0,1.1), legend=False,
               title='Informative: ' + str(pmean.index[ratio]),
               color=sns.color_palette(n_colors=2))
        
filename = os.path.join(output_folder, 'fs_benchmark_summary_r0_5.pdf')
plt.tight_layout()
fig.savefig(filename, dpi=300, bbox_inches='tight')
plt.clf()