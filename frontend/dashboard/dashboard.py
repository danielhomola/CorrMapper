import numpy as np
import os
import pandas as pd
import pickle
import scipy as sp
import traceback
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from frontend import app, models
from frontend.view_functions import get_study_folder
from backend.utils.check_uploaded_files import open_file
from write_dashboard_js import write_dashboard_js as write_js


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------

def dashboard(study_id, window_dims):
    """
    The main function preparing the json file for the dashboard and writing the
    custon JS file that does the interactive charting.
    """
    # -------------------------------------------------------------------------
    # DASHBOARD.JSON
    # -------------------------------------------------------------------------

    # get basic variables for study
    study = models.Studies.query.get(study_id)
    study_folder = get_study_folder(study_id)
    dashboard_folder = os.path.join(study_folder, 'dashboard')
    metadata_file = "dash_" + study.metadata_file
    datasets = [study.dataset1]
    autocorr = study.autocorr
    if not autocorr:
        datasets.append(study.dataset2)

    # handle missing values and dates
    try:
        meta_path = os.path.join(study_folder, metadata_file)
        meta, _ = open_file(meta_path, header=[0, 1])
        # sort column MultiIndex so we can do awesome multi-indexing
        meta = meta.sortlevel((0,1), axis=1)
        meta, min_med_max = missing_metadata(meta)
    except:
        app.logger.error('Variable checking for metadata explorer failed for '
                         'study: %d\n%s' % (study_id, traceback.format_exc()))
        return False, None, None

    # calculate PCA scores for dataset(s)
    try:
        datasets[0], _ = open_file(os.path.join(study_folder, datasets[0]))
        if not autocorr:
            datasets[1], _ = open_file(os.path.join(study_folder, datasets[1]))
        num_comp = app.config['NUM_PCA_COMPONENTS']
        meta, min_med_max, pc_var = calculate_PCA(datasets, meta, min_med_max, num_comp)
    except:
        app.logger.error('PCA calculation in metadata explorer failed for '
                         'study: %d\n%s' % (study_id, traceback.format_exc()))
        return False, None, None

    # save dashboard.json
    meta_json = meta.copy(deep=True)
    meta_json.columns = meta_json.columns.droplevel(1)
    meta_json.to_json(os.path.join(dashboard_folder, 'dashboard.json'),
                      orient='records', date_format='iso')

    # -------------------------------------------------------------------------
    # DASHBOARD.JS
    # -------------------------------------------------------------------------

    try:
        chart_dims = layout(window_dims, autocorr)
        charts, dash_vars = write_js(dashboard_folder, autocorr, meta,
                                     min_med_max, pc_var, chart_dims, num_comp)
    except:
        app.logger.error('Writing dashboard.js failed in study: %d\n%s'
                         % (study_id, traceback.format_exc()))
        return False, None, None
    return True, charts, dash_vars

# -----------------------------------------------------------------------------
# DASHBOARD FUNCTIONS: HANDLING METADATA, CALCULATING PCA, LAYOUTING
# -----------------------------------------------------------------------------

def missing_metadata(meta):
    """
    Go through each column of the metadata file and handle missing values and dates
    """
    # this dict holds the nan aware min, median and max of all continuous cols
    min_med_max = {}
    metacols = meta.columns.get_level_values(0)
    metatype = meta.columns.get_level_values(1)

    # take care of missing values for cat, patient with "Missing"
    if np.any(metatype == 'cat'):
        # enforce String type for categorical variables if they are numbers
        cat_cols = meta.loc[:, (slice(None), "cat")]
        not_str = (cat_cols.dtypes != "object").values
        if np.any(not_str):
            meta.update(cat_cols.iloc[:, not_str].fillna("Missing").astype(str))
        # missing values
        meta.update(meta.loc[:, (slice(None), 'cat')].fillna("Missing"))

    if np.any(metatype == 'patient'):
        meta.update(meta.loc[:, (slice(None), 'patient')].fillna("Missing"))

    # take care of missing values for cont with 2x max
    if np.any(metatype == 'con'):
        for col in metacols[metatype == 'con']:
            min_med_max[col] = {
                'min': meta[col].min().values[0] - abs(meta[col].min().values[0]) * .05,
                'median': meta[col].median().values[0],
                'max': meta[col].max().values[0] + abs(meta[col].max().values[0]) * .05
            }
        fakemax = meta.loc[:, (slice(None), 'con')].max() * 2
        meta.update(meta.loc[:, (slice(None), 'con')].fillna(fakemax))


    # take care of missing values for date: set it to 2200 in the future
    if np.any(metatype == 'date'):
        date_col = metacols[metatype == 'date']
        date = pd.to_datetime(meta[date_col].values.ravel(), errors='coerce',
                              infer_datetime_format=True)
        # calculate min, median and max date on the rows where we have date
        non_nat = date[~pd.isnull(date)]
        min_date = non_nat.min()
        delta_date = pd.to_timedelta((non_nat - min_date).astype('m8[ms]').to_series().median(), unit='ms')
        median_date = min_date + delta_date
        max_date = non_nat.max()
        min_med_max['Date'] = {
            'min': min_date,
            'median': median_date,
            'max': max_date
        }

        # delete user's date columns and insert new ones, put 2200 for missing dates
        meta.drop(date_col, 1, inplace=True)
        meta.insert(meta.shape[1], ('Date', 'date'), pd.Series(date).fillna(pd.Timestamp('22000101')).values)

        # add year
        date = pd.to_datetime(meta.loc[:, ('Date', 'date')])
        year = date.dt.year
        meta.insert(meta.shape[1], ('Year', 'date'), year)

        # add month
        months = {
            '1': 'Jan',
            '2': 'Feb',
            '3': 'Mar',
            '4': 'Apr',
            '5': 'May',
            '6': 'Jun',
            '7': 'Jul',
            '8': 'Aug',
            '9': 'Sep',
            '10': 'Oct',
            '11': 'Nov',
            '12': 'Dec',
        }
        months = [months[str(m)] for m in date.dt.month.values]
        meta.insert(meta.shape[1], ('Month', 'date'), months)

        # add weekday
        weekdays = {
            '0': 'Mon',
            '1': 'Tue',
            '2': 'Wed',
            '3': 'Thu',
            '4': 'Fri',
            '5': 'Sat',
            '6': 'Sun'
        }
        weekdays = [weekdays[str(m)] for m in date.dt.dayofweek.values]
        meta.insert(meta.shape[1], ('Day', 'date'), weekdays)

    # add ID columns
    meta.insert(meta.shape[1], ('CorrMapperID', 'id'), range(meta.shape[0]))
    meta.insert(meta.shape[1], ('ID', 'id'), meta.index.values)
    return meta, min_med_max


def calculate_PCA(datasets, meta, min_med_max, num_comp):
    """
    Returns the metadata DataFrame with update PC1 and PC2 coordinates for
    (both) dataset(s) and a pc_var dictionary with the explained variance.
    """
    pca = PCA(n_components=num_comp)
    pc_var = {}
    for pc in range(num_comp):
        meta.insert(meta.shape[1], ('dataset1_pc' + str(pc + 1), 'pca'), np.zeros(meta.shape[0]))
        meta.insert(meta.shape[1], ('dataset2_pc' + str(pc + 1), 'pca'), np.zeros(meta.shape[0]))

    for i, dataset in enumerate(datasets):
        # only use samples for the PCA for which we have metadata: rows in both
        dataset_meta_ind = dataset.index.intersection(meta.index)
        # save rows that are only in metadata
        meta_not_dataset_ind = set(meta.index) - set(dataset_meta_ind)
        X = dataset.loc[dataset_meta_ind]
        # mean center
        X -= X.mean(axis=0)
        scores = pca.fit_transform(X)
        # save scores to meta
        dataset_count = 'dataset' + str(i + 1)
        # dicts to hold PCs and their variance
        min_med_max[dataset_count] = {}
        pc_var[dataset_count] = {}
        for pc in range(num_comp):
            meta.loc[dataset_meta_ind, (dataset_count + '_pc' + str(pc + 1), 'pca')] = scores[:, pc]
            # save min and max values of pca scores
            min_med_max[dataset_count]['pc' + str(pc + 1) + '_min'] = np.min(scores[:, pc])
            min_med_max[dataset_count]['pc' + str(pc + 1) + '_max'] = np.max(scores[:, pc])
            # set missing rows to twice the max (dc.js won't show these in scatter)
            meta.loc[meta_not_dataset_ind, (dataset_count + '_pc' + str(pc + 1), 'pca')] = scores[:, pc].max() * 2

            # save pcs dict
            pc_var[dataset_count]['pc' + str(pc + 1)] = pca.explained_variance_ratio_[pc] * 100
    return meta, min_med_max, pc_var


def layout(window_dims, autocorr):
    """
    Based on user's window width calculates the sizes of the dashboard charts.
    """
    # basic variables
    window_width = int(window_dims[0])
    window_height = int(window_dims[1])
    chart_sizes = {}
    if window_width > 1800:
        charts_per_row = 6
        chart_sizes['charts_per_row'] = charts_per_row
        chart_sizes['pie_chart_radius'] = 30
        padding = 150
    else:
        charts_per_row = 3
        chart_sizes['charts_per_row'] = charts_per_row
        chart_sizes['pie_chart_radius'] = 35
        padding = 100
    half_width = window_width / 2

    # chart sizes
    chart_sizes['pie_chart_width'] = (half_width - padding) / charts_per_row
    return chart_sizes

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR SAVING DASHBAORDS AND LOADING THEM
# -----------------------------------------------------------------------------

def save_obj(obj, path):
    """
    Saves an object using pickle
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    """
    Loads a pickled object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_dashboard(study_id):
    """
    Loads two essential dicts for the dashboard that was generated by
    write_dashboard_js.py and that's needed for rendering dashboard.html
    """
    study_folder = get_study_folder(study_id)
    dashboard_folder = os.path.join(study_folder, 'dashboard')
    charts = load_obj(os.path.join(dashboard_folder, 'dashboard_charts.pkl'))
    dash_vars = load_obj(os.path.join(dashboard_folder, 'dashboard_dash_vars.pkl'))
    return charts, dash_vars


def save_dashboard(study_id, charts, dash_vars):
    """
    Saves two essential dicts for the dashboard that was generated by
    write_dashboard_js.py and that's needed for rendering dashboard.html
    """
    study_folder = get_study_folder(study_id)
    dashboard_folder = os.path.join(study_folder, 'dashboard')
    save_obj(charts, os.path.join(dashboard_folder, 'dashboard_charts.pkl'))
    save_obj(dash_vars, os.path.join(dashboard_folder, 'dashboard_dash_vars.pkl'))


def check_dashboard(study_id):
    """
    Returns True if dashboard main function needs to be run to generate
    all that goes into dashboard.html, and False if everything is already generated.
    """
    study_folder = get_study_folder(study_id)
    dashboard_folder = os.path.join(study_folder, 'dashboard')
    to_check = ['dashboard.js', 'dashboard.json', 'dashboard_charts.pkl', 'dashboard_dash_vars.pkl']
    check_result = False
    for file in to_check:
        if not os.path.exists(os.path.join(dashboard_folder, file)):
            check_result = True
            break
    return check_result
