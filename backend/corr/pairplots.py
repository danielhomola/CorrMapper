import os
from shutil import copy

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from backend.utils.check_uploaded_files import open_file


def get_png_name(f1, f2):
    """
    Given two feature names it generates a unique commutative file name from
    them. E.g.: f1="asd", f2="fgh" will get mapped to unicode and added up one
    letter at a time.
    """
    lf1 = len(f1)
    lf2 = len(f2)
    # make them equal length
    if lf1 > lf2:
        f2 += 'a' * (lf1-lf2)
    elif lf2 > lf1:
        f1 += 'a' * (lf2-lf1)
    # map each letter to unicode and add them up across the two feature names
    filename = ''.join(map(str, [ord(i[0]) + ord(i[1]) for i in zip(f1, f2)]))
    return filename + ".png"


def generate_pair_plots(params, rs, p_vals, datasets, p):
    """
    Generate a scatter plots for each pair of variables in a correlation matrix.
    """
    # setup plots
    sns.set_style("whitegrid", {"grid.color": ".95"})

    # make folders in output img folder (this will be zipped)
    output_img_folder = params["output_img_folder"]
    dataset1_folder = os.path.join(output_img_folder, 'dataset1')
    if not os.path.exists(dataset1_folder):
        os.makedirs(dataset1_folder)
    if not params['autocorr']:
        dataset2_folder = os.path.join(output_img_folder, 'dataset2')
        if not os.path.exists(dataset2_folder):
            os.makedirs(dataset2_folder)
        dataset1_2_folder = os.path.join(output_img_folder, 'dataset1_2')
        if not os.path.exists(dataset1_2_folder):
            os.makedirs(dataset1_2_folder)

    # if fs open metadata file - we'll colour each point by it
    if params['fs']:
        metadata_name = params['fs_cols']
        path = os.path.join(params['study_folder'], params['metadata_file'])
        y, _ = open_file(path)
        y = y[metadata_name]
        y_ = y.iloc[1:]
        meta_type = y.iloc[0]

        # find intersecting samples, filter down y and add it to X's end
        ind = datasets.index.intersection(y_.index)
        datasets = datasets.loc[ind]
        y_ = y_.loc[ind].values
        datasets.insert(datasets.shape[1], metadata_name, y_)
    else:
        # no fs, we cannot colour the scatter plots by anything
        meta_type = None

    # heatmap col and row names are truncated in utils.recode_rowcol_names() so
    # we need to truncate them here as well so they match the feature names
    threshold = 12

    # loop through the lower triangular part of the R matrix and plot each pair
    lower_tri_ind = np.tril_indices(rs.shape[1], k=-1)
    for i in xrange(lower_tri_ind[0].shape[0]):
        y_loc = lower_tri_ind[0][i]
        x_loc = lower_tri_ind[1][i]
        r_val = rs[x_loc, y_loc]
        p_val = p_vals[x_loc, y_loc]
        if r_val != 0:
            x_var = datasets.columns[x_loc]
            y_var = datasets.columns[y_loc]
            suptitle = r"$R^2$:%s, p-value:%s" % ("{:0.3f}".format(r_val),
                                                  "{:0.6f}".format(p_val))
            # categorical metadata variable plots
            if meta_type == "cat":
                g = sns.lmplot(x=x_var, y=y_var, hue=metadata_name,
                               data=datasets, size=4, aspect=1, ci=68,
                               scatter_kws={'alpha': 0.6, 's': 40},
                               line_kws={'alpha': 0.5, 'linewidth': 1.5})
                g.fig.suptitle(suptitle)
            # # continuous metadata variable plots
            elif meta_type == "con":
                cmap = sns.cubehelix_palette(as_cmap=True)
                g, ax = plt.subplots(figsize=(5, 5))
                points = ax.scatter(datasets[x_var], datasets[y_var],
                                    c=y_, cmap=cmap)
                clb = g.colorbar(points)
                clb.ax.set_title(metadata_name)
                plt.title(suptitle, loc='left')
                plt.xlabel(x_var)
                plt.ylabel(y_var)
            # no metadata
            else:
                g = sns.lmplot(x=x_var, y=y_var, data=datasets, size=4, aspect=1)
                g.fig.suptitle(suptitle)

            # find out which correlation sub-matrix the plot belongs to
            if x_loc < p and y_loc < p:
                dataset_folder = dataset1_folder
            elif x_loc >= p and y_loc >= p:
                dataset_folder = dataset2_folder
            else:
                dataset_folder = dataset1_2_folder

            # save image into output and analysis folders too
            filename = x_var[:threshold] + "_" + y_var[:threshold] + ".png"
            plt_output_path = os.path.join(dataset_folder, filename)
            g.savefig(plt_output_path)
            # save it into img_folder with commutative name
            filename = get_png_name(x_var[:threshold], y_var[:threshold])
            plt_img_path = os.path.join(params["img_folder"], filename)
            copy(plt_output_path, plt_img_path)
            # close current plotting window so we don't use too much memory
            plt.close()
