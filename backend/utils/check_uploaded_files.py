"""
These functions make sure that only studies with correctly formatted files are
saved and allowed to be analysed.
"""


import os
import re
import numpy as np
import traceback
import pandas as pd
import shutil
from frontend import app
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename
from frontend.view_functions import get_user_folder

# -----------------------------------------------------------------------------
# CHECK FILES MAIN FUNCTION
# -----------------------------------------------------------------------------

def check_files(user_data_folder, files_dict, form):
    """
    Checks if the uploaded files have the correct format
    """
    # -------------------------------------------------------------------------
    # SET UP OF VARIABLES
    format_errors = {}
    # this will show a top panel on the form, informing the user
    format_errors['misformatted'] = ''
    annotation = bool(form.annotation.data)
    autocorr = not (form.autocorr.data)
    fs = (form.fs.data)
    study_folder = secure_filename(form.study_name.data)

    # -------------------------------------------------------------------------
    # CHECKING DATASETS
    # -------------------------------------------------------------------------
    datasets = ['dataset1']
    samples = []
    probes = []
    if not autocorr:
        datasets.append('dataset2')

    for dataset in datasets:
        # check if we can open the file
        try:
            dataset_path = os.path.join(user_data_folder, files_dict[dataset])
            df, sep = open_file(dataset_path)
        except:
            format_errors[dataset] = ['Could not open this file.']
            clear_up_study(study_folder)
            return False, format_errors

        # check if we have enough numeric columns in all data files
        df_numeric = df.select_dtypes(include=[np.number])

        # save row and col names of datasets
        samples.append(df_numeric.index)
        probes.append(df_numeric.columns)
        min_numeric = app.config['MINIMUM_FEATURES']
        if df_numeric.shape[1] < min_numeric:
            format_errors[dataset] = ['Datasets must have at least %s numeric'
                                     ' columns.' % min_numeric]
            clear_up_study(study_folder)
            return False, format_errors

        # check dataset dimensions
        if (df.shape[0] > app.config['MAX_SAMPLES'] or
            df.shape[1] > app.config['MAX_FEATURES']):
            format_errors[dataset] = ['The dimensions of this dataset exceed '
                                      'the supported maximum.']
            clear_up_study(study_folder)
            return False, format_errors

        # impute missing values with median
        df_numeric = df_numeric.fillna(df_numeric.median())

        # save imputed, all-numeric dataset
        df_numeric.to_csv(dataset_path, sep=sep)

    # if we have two datasets check if we have enough intersecting columns
    min_intersecting = app.config['INTERSECTING_SAMPLES']
    if not autocorr:
        samples_intersect = np.intersect1d(samples[0], samples[1])
        if samples_intersect.shape[0] <= min_intersecting:
            format_errors[dataset] = ['Datasets must have at least %s shared '
                                     'samples.' % min_intersecting]
            clear_up_study(study_folder)
            return False, format_errors

    # -------------------------------------------------------------------------
    # CHECKING ANNOTATIONS
    # -------------------------------------------------------------------------
    annotations = []
    if annotation:
        annotations.append('annotation1')
        # convert list of column names into dict
        probes0 = list(probes[0].values)
        probes[0] = dict(zip(probes0,[1] * len(probes0)))
        species = form.species.data.replace(' ','_')
        if not autocorr:
            annotations.append('annotation2')
            probes1 = list(probes[1].values)
            probes[1] = dict(zip(probes1,[1] * len(probes1)))

        for i, annotation in enumerate(annotations):
            # check if we can open the file
            try:
                annotation_path = os.path.join(user_data_folder,
                                               files_dict[annotation])
                df, sep = open_file(annotation_path)
            except:
                format_errors[annotation] = ['Could not open this file.']
                clear_up_study(study_folder)
                return False, format_errors

            # check annotation files
            try:
                status, errors = check_annotation_files(annotation_path,
                                                        probes[i], species)
                if not status:
                    format_errors[annotation] = errors
                    clear_up_study(study_folder)
                    return False, format_errors
            except:
                format_errors[annotation] = ['Misformatted annotation file.']
                clear_up_study(study_folder)
                return False, format_errors

    # -------------------------------------------------------------------------
    # CHECKING METADATA
    # -------------------------------------------------------------------------
    metadata_cols = None
    if fs:
        # check if we can open the file
        try:
            meta = 'metadata_file'
            metadata_path = os.path.join(user_data_folder, files_dict[meta])
            df, sep = open_file(metadata_path)
        except:
            format_errors[meta] = ['Could not open this file.']
            clear_up_study(study_folder)
            return False, format_errors

        # create path for metadata file that'll be used for dashboard
        dash_meta = "dash_" + files_dict[meta]
        dash_metadata_path = os.path.join(user_data_folder, dash_meta)

        # check number of columns of dashboard file, resize if necessary
        max_metadata_cols = app.config['MAX_METADATA_COLUMNS']
        if df.shape[1] > max_metadata_cols:
            df = df.iloc[:,:max_metadata_cols]

        # check if the columns all have categorical, continuous or date labels
        categorical = ['categorical', 'cat', 'categ', 'ca']
        continuous = ['continuous', 'con', 'cont', 'co']
        date = ['date', 'time']
        patient = ['sample', 'patient']
        reserved_col_names = ['Year', 'Month', 'Day', 'CorrMapperID',
                              'dataset1_pc1', 'dataset1_pc2', 'dataset2_pc1',
                              'dataset2_pc2', 'all', 'ID']
        # save types of cols
        metadata_types = df.iloc[0,:max_metadata_cols].values
        # get rid of all whitespaces in variable names
        df = df.rename(columns=lambda x: re.sub(r"\s+", '_', x))
        # if we have columns that are legit, save them to database
        metadata_cols = []

        # create copy for dashboard which won't check number of samples of cats
        df_dash = df.copy(deep=True)

        # go through each column and check it
        for i, metadata_type in enumerate(metadata_types):
            m = metadata_type.lower()
            if df.columns[i] in reserved_col_names:
                df.rename(columns={df.columns[i]:df.columns[i] + '_user'},
                          inplace=True)
                df_dash.rename(columns={df.columns[i]: df.columns[i] + '_user'},
                               inplace=True)
            if m not in (categorical + continuous + date + patient):
                format_errors[meta] = ['Cannot recognise  the type of some '
                                       'metadata variables.']
                clear_up_study(study_folder)
                return False, format_errors
            else:
                # if categorical convert it to numbers
                if m in categorical:
                    df.iloc[0, i] = 'cat'
                    df_dash.iloc[0, i] = 'cat'
                    # check if we have enough samples in all class
                    meta = df.iloc[1:, i].dropna()
                    check, meta = check_cat_meta(samples, meta, app.config['MIN_SAMPLE_PER_CLASS'])
                    if check:
                        # overwrite meta with updated version
                        df.loc[meta.index, df.columns[i]] = meta
                        metadata_cols.append(df.columns[i])
                elif m in continuous:
                    # check continuous / numeric features
                    df.iloc[0, i] = 'con'
                    df_dash.iloc[0, i] = 'con'
                    try:
                        np.float64(df.iloc[1:, i].values)
                    except:
                        format_errors[meta] = ['Column %s has non-numeric values.' % df.columns[i]]
                        clear_up_study(study_folder)
                        return False, format_errors
                    metadata_cols.append(df.columns[i])
                elif m in date:
                    # check dates
                    df.iloc[0, i] = 'date'
                    df_dash.iloc[0, i] = 'date'
                    try:
                        pd.to_datetime(df.iloc[1:,i].values,
                                       infer_datetime_format=True)
                    except:
                        format_errors[meta] = ['Cannot recognise date format.']
                        clear_up_study(study_folder)
                        return False, format_errors
                elif m in patient:
                    df.iloc[0, i] = 'patient'
                    df_dash.iloc[0, i] = 'patient'

        # safeguard return value if all dashboard cols are bad
        if len(metadata_cols) == 0:
            metadata_cols = None

        # check we have enough shared samples
        if not autocorr:
            samples_intersect2 = np.intersect1d(samples_intersect, df.index)
        else:
            samples_intersect2 = np.intersect1d(samples[0], df.index)
        if samples_intersect2.shape[0] <= min_intersecting:
            format_errors[dataset] = ['Datasets must have at least %s shared '
                                     'samples.' % min_intersecting]
            clear_up_study(study_folder)
            return False, format_errors

        # save metadata files
        df.to_csv(metadata_path, sep=sep)
        df_dash.to_csv(dash_metadata_path, sep=sep)

    # everything went fine, every file is checked
    return True, metadata_cols

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def clear_up_study(study_folder):
    """
    Deletes the uploaded files if they are mis-formatted.
    """
    user_folder = get_user_folder()
    folder_to_delete = os.path.join(user_folder, study_folder)
    if os.path.exists(folder_to_delete):
        shutil.rmtree(folder_to_delete)


def open_file(file_path, **kwargs):
    """
    Opens files based on their file extension in pandas
    """
    filename, extension = os.path.splitext(file_path)
    if extension == '.txt':
        sep = '\t'
    else:
        sep = ','
    file = pd.read_csv(file_path, sep=sep, index_col=0, **kwargs)
    return file, sep

# -----------------------------------------------------------------------------
# CHECK ANNOTATION FILES
# -----------------------------------------------------------------------------

def check_annotation_files(annotation_path, genomic_elements, species):
    """
    Checks the annotation file for the genomic elements.

    Header must be of equal length or longer than the columns.

    Discards all lines that don't have a valid (i.e. + integer) start and end
    position and the ones that are on chromosomes we don't have in the binFile.

    Also each genomic element's start and end position should be within the
    chromosome otherwise it's discarded.
    """
    annotation_discarded_header = False
    errors = []

    # open original annotation file and create checked version
    filename, extension = os.path.splitext(annotation_path)
    _, sep = open_file(annotation_path)
    annotation_checked_path = filename + '_checked' + extension
    annotation_discarded_path = filename + '_discarded' + extension
    try:
        annotation = open(annotation_path, 'r')
        annotation_checked = open(annotation_checked_path, 'w')
        annotation_discarded = open(annotation_discarded_path, 'w')
    except IOError:
        errors.append('Cannot open annotation file')
        return False, errors

    # get valid chromosome names and  lengths
    try:
        bins_folder = os.path.join(app.config['BACKEND_FOLDER'], 'bins')
        chromo_file = species + '__chromosomes.txt'
        chromo_path = os.path.join(bins_folder, chromo_file)
        chromo = pd.read_table(chromo_path, index_col=0)
        chromosomes = pd.unique(chromo.index)
        chromo_lengths = chromo.to_dict(orient='dict')['Length']
    except (IOError, KeyError):
        errors.append('Cannot open chromosome file, please contact the authors.')
        return False, errors

    # read through annotation file
    for i, line in enumerate(annotation):
        line_array = line.strip().split(sep)
        # make sure the header is as long as the following lines
        if i == 0:
            first_cols = sep.join(['ProbeID','Chromosome','Start','End'])
            if len(line_array) > 4:
                rest_of_header = sep.join(line_array[4:]) + '\n'
            else:
                rest_of_header = '\n'
            annotation_checked.write(first_cols + sep + rest_of_header)
        # check the body of the annotation
        else:
            try:
                p = str(line_array[0])
                c = str(line_array[1])
                s = int(line_array[2])
                e = int(line_array[3])
                maxl = chromo_lengths[c]
                if (len(line_array) >= 4 and
                    c in chromosomes and p in genomic_elements and
                    float(s).is_integer() and float(e).is_integer() and
                    s > 0 and e > 0 and s < maxl and maxl > e > s):
                    annotation_checked.write(line)
                else:
                    raise ValueError('')
            except (ValueError, KeyError):
                if not annotation_discarded_header:
                    w = ('The following line(s) were discarded'
                        ' for one of the following reasons:\n1. Had less than '
                        '4 columns.\n2. Had a chromosome name that did not '
                        'match with the chromosomes of the organism.\n3. Had a'
                        ' start or end position lower than 0.\n4. Had a higher'
                        ' start than end position.\n5. Had an end position '
                        'that is higher than what we have for the chromosome.'
                        '\n\n' + 'Line %s - ' % i + line)
                    annotation_discarded.write(w)
                    annotation_discarded_header = True
                else:
                     annotation_discarded.write('Line %s - ' % i + line)
    annotation.close()
    annotation_checked.close()
    annotation_discarded.close()

    return True, errors

# -----------------------------------------------------------------------------
# CHECK CATEGORICAL METADATA COLUMN
# -----------------------------------------------------------------------------

def check_cat_meta(samples, meta, min_samples_per_class):
    """
    This is called for each categorical metadata variable. We check if we have
    enough samples per each level of the categorical variable.

    If all levels have less than what we accept we exclude the column from the
    analysis, but if we have fewer samples in a few levels AND we still got
    two levels which have sufficient samples, then we just set the insufficient
    levels to nan so they'll be filtered in the feature selection step.

    This function is not meant to turn string valued levels of categorical vars
    into numbers. LabelEncoder is only used to count the levels. When we do FS
    later, we will transform the labels, otherwise we need them in string format
    for the dashboard and pairwise plots.
    """
    check = True
    to_nan = {}
    for sample in samples:
        # take intersection with dataset index
        intersection = np.intersect1d(sample, meta.index)

        meta_inter = meta.loc[intersection]
        le = LabelEncoder()
        le_fit = le.fit(meta_inter.values)
        meta_transformed = le.fit_transform(meta_inter.values)
        # check if we have enough samples in each class
        class_counts = np.bincount(meta_transformed)
        # only allow feature selection on cat cols that are sensible
        if np.all(class_counts < min_samples_per_class):
            check = False
        # no variation in metadata
        elif len(class_counts) == 1:
            check = False
        # we only have one level left with enough samples, we have to discard
        elif np.sum(class_counts > min_samples_per_class) == 1:
            check = False
        # register the levels we need to set to nan
        else:
            too_few = np.where(class_counts < min_samples_per_class)[0]
            for level in too_few:
                to_nan[le_fit.classes_[level]] = 1
    # check if it makes sense to set levels to nan, i.e. at least 2 levels left
    le_fit = le.fit(meta.values)
    if le_fit.classes_.shape[0] - len(to_nan.keys()) > 1:
        for i in to_nan.keys():
            meta.loc[meta == i] = np.nan
    else:
        check = False

    return check, meta
