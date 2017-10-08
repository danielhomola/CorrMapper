import os

from write_network import write_network as wn
from write_tables import write_tables as wt
from backend.utils.check_uploaded_files import open_file


def genomic_main(params):
    """
    Wrapper for write_vis_Genomic which writes JS files for the vis_genomic.
    """

    # DATA 1
    # open necessary files
    path = os.path.join(params['study_folder'], params['annotation1'])
    annot1, _ = open_file(path)
    annot = [annot1, annot1]
    path = os.path.join(params['output_folder'], params['r_dataset1'])
    r, _ = open_file(path)
    path = os.path.join(params['output_folder'], params['p_dataset1'])
    p, _ = open_file(path)

    # write network, table files and variables
    bin2label, row2bin, _ = wn(params, 'dataset1', annot, r, True)
    wt(params, 'dataset1', annot1, r, p, row2bin, bin2label, axis=1)
    wt(params, 'dataset1', annot1, r, p, row2bin, bin2label, axis=0)

    if not params['autocorr']:
        # DATA 2
        path = os.path.join(params['study_folder'], params['annotation2'])
        annot2, _ = open_file(path)
        annot = [annot2, annot2]
        path = os.path.join(params['output_folder'], params['r_dataset2'])
        r, _ = open_file(path)
        path = os.path.join(params['output_folder'], params['p_dataset2'])
        p, _ = open_file(path)

        # write network, table files and variables
        _, row2bin, _ = wn(params, 'dataset2', annot, r, True)
        wt(params, 'dataset2', annot2, r, p, row2bin, bin2label, axis=1)
        wt(params, 'dataset2', annot2, r, p, row2bin, bin2label, axis=0)

        # DATA 1_2
        annot = [annot1, annot2]
        path = os.path.join(params['output_folder'], params['r_dataset1_2'])
        r, _ = open_file(path)
        path = os.path.join(params['output_folder'], params['p_dataset1_2'])
        p, _ = open_file(path)

        # write network, table files and variables
        _, row2bin, col2bin = wn(params, 'dataset1_2', annot, r)
        wt(params, 'dataset1_2', annot1, r, p, row2bin, bin2label, axis=1)
        wt(params, 'dataset1_2', annot2, r, p, col2bin, bin2label, axis=0)
    return params
