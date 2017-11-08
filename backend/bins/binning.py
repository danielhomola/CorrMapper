"""
Methods for binning.

Checking chromosome files, creating binning files and binning genomic elements
based on chromosome files.

In the current version of CorrMapper we not allow users to upload chromo files.
They are only allowed to choose from the predifened ones. But the
check_chromo_file and bin_maker functions are kept at the end of this module
for creating new binnings if the users request it.
"""

import inspect
import numpy as np
import os
import pandas as pd

from backend.utils.check_uploaded_files import open_file


def bin_genomic_elements_main(params):
    """
    Wrapper to deal with the binning of both annot files if there are two.
    """

    # bin annotation1 file and update the params dict

    params['annotation1'], params['dataset1'] = bin_genomic_elements(params,
                                                    'annotation1', 'dataset1')
    # if we have two files bin the second as well
    if not params['autocorr']:
        params['annotation2'], params['dataset2'] = bin_genomic_elements(params,
                                                    'annotation2', 'dataset2')
    return params


def bin_genomic_elements(params, annotation, data):
    """
    Bins genomic elements based on the supplied _bins.txt file.

    For every element it requires a chromosome str, and start and end pos.
    """
    study_folder = params['study_folder']
    bin_file = params['bin_file']
    annot_file = params[annotation]
    # open bin and checked annotation file
    bins = pd.read_table(bin_file)
    annot, sep = open_file(os.path.join(study_folder, annot_file))

    # bin GEs
    # extract start of chromosomes
    chromo_starts = {}
    for c in list(np.where(bins.ChromoStart != 0)[0]):
        chromo_starts[bins.Chromosome[c]] = bins.ChromoStart[c]

    # convert relative start and end positions to absolute
    starts = ([chromo_starts[x] for x in annot.Chromosome] + annot.Start).values
    ends = ([chromo_starts[x] for x in annot.Chromosome] + annot.End).values
    starts_binned = np.digitize(starts, bins.Absolute)
    ends_binned = np.digitize(ends, bins.Absolute)

    # check if any genomic element got binned outside
    diff_bin_pos = (ends_binned - starts_binned)

    # inTwoBins
    in_two_bins = np.where(diff_bin_pos == 1)[0]
    absolute_in_between = (bins.Absolute[ends_binned[in_two_bins] - 1]).values
    larger_in_end = in_two_bins[np.where(
        absolute_in_between - starts[in_two_bins] <
        ends[in_two_bins] - absolute_in_between)[0]]
    # in more than two bins - really rare: we just assign the bin inbetween
    in_more_than_two_bins = np.where(diff_bin_pos > 1)[0]

    # add final bin annotation to annotFile
    starts_binned[larger_in_end] = ends_binned[larger_in_end]
    starts_binned[in_more_than_two_bins] += 1
    annot.insert(3, 'Bin', starts_binned)

    # write binned version of annotationFile
    file_name, file_extension = os.path.splitext(annot_file)
    annot_binned = file_name.replace('checked', 'binned') + file_extension
    annot.to_csv(os.path.join(study_folder, annot_binned), sep=sep)

    # rewrite data files to only contain genomic elements that got binned
    data_file = params[data]
    file_name, file_extension = os.path.splitext(data_file)
    data, sep = open_file(os.path.join(study_folder, data_file))
    data_binned = file_name + ('_binned') + file_extension
    # columns of the data file are the rows of the annotation file
    data = data[annot.index]
    data.to_csv(os.path.join(study_folder, data_binned), sep=sep)

    return annot_binned, data_binned


# -----------------------------------------------------------------------------
# THESE FUNCTIONS MAKE BIN FILE FROM NEW CHROMO FILE
# -----------------------------------------------------------------------------


def check_chromo_file(chromo_file):
    """
    Checks the format of chromosome file, before the bin_maker proceeds.
    """

    o = open(chromo_file)
    for i, line in enumerate(o):
        line_array = line.strip().split('\t')

        # check header
        if i == 0:
            if len(line_array) != 2:
                raise ValueError('Header should be: Chromosome\tLength' +
                             inspect.stack()[0][3])

        # check lines
        else:
            if len(line_array) == 2:
                try:
                    v = int(line_array[1])
                    if not float(v).is_integer() or v < 0:
                        raise ValueError('Must be positive, integer')
                except ValueError:
                    raise ValueError('Misformatted binning file. Chromosome '
                                 'lengths should be positive integers.\n' +
                                 traceback.format_exc())
            else:
                raise ValueError('Misformatted binning file. Each line should '
                                 'be: Chromosome name\t Chromosome length.' +
                                inspect.stack()[0][3])
    o.close()


def bin_maker(chromo_file, num_of_bins):
    """
    Based on a _chromosome.txt file, makes _bins.txt file for the species.

    First this file is checked as annotation files.
    """

    # check the format of the chromo file
    check_chromo_file(chromo_file)

    # binning file is fine, let's make _bins.txt file
    try:
        chromo = pd.read_table(chromo_file, index_col=0)
    except IOError:
        raise IOError('Cannot open chromosome file.\n'
                      + traceback.format_exc())

    try:
        bin_len = chromo.Length.sum() / num_of_bins
        chromo.insert(2, 'num_of_bins_float', chromo.Length / bin_len)
        chromo.insert(3, 'num_of_bins_int', (chromo.Length / bin_len).astype(int))

        # calculate which chromosomes need an extra bin
        num_of_half_bins = num_of_bins - sum(chromo.num_of_bins_int)
        fractions = (chromo.num_of_bins_float - chromo.num_of_bins_int).values
        chroms_needing_extra = np.argsort(fractions)[::-1][0:num_of_half_bins]
        chromo.num_of_bins_int[chroms_needing_extra] += 1

        # go through chromosomes and make the bins
        bin_file = chromo_file.replace('_chromosomes.txt', '_bins.txt')
        o = open(bin_file, 'w')
        o.write('Chromosome\tStart\tEnd\tBin\tAbsolute\tChromoStart\n')
        # actual bin
        bin_count = 1
        # holds absolute bin position in the genome
        bin_absolute = 1

        for c in range(chromo.shape[0]):
            bin_num = int(chromo.iloc[c, 3])
            for b in range(bin_num):
                bin_start = b * bin_len + 1
                chromo_start = 0

                # bin number dependent variables
                if b == 0:
                    chromo_start = bin_absolute
                if b < bin_num - 1:
                    bin_end = (b + 1) * bin_len + 1
                else:
                    bin_end = chromo.iloc[c, 1] + 1

                # write new line
                to_write = [int(chromo.iloc[c, 0]), int(bin_start), int(bin_end),
                            int(bin_count), int(bin_absolute), int(chromo_start)]
                o.write('\t'.join(map(str, to_write)) + '\n')
                bin_count += 1

                # increment bin absolute
                if b == bin_num - 1:
                    bin_absolute += bin_end - bin_start + 1
                else:
                    bin_absolute += bin_len
        o.close()
        return bin_file
    except Exception:
        raise ValueError(traceback.format_exc())
