"""
These are essential helper functions for the upload, analysis and profile views.
"""

import datetime
import json
import os

from flask import request
from flask_security import current_user
from werkzeug.utils import secure_filename

from . import app, db, models

# -----------------------------------------------------------------------------
# UPLOAD - HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_form(form, files):
    """
    Merges files and form values of request, because flask separates them.
    """
    form_dict = {}
    for k, v in request.values.items():
        form_dict[k] = v
    for k, v in request.files.items():
        form_dict[k] = v
    return form_dict


def save_study(form, files):
    """
    Creates study folder, saves uploaded files, checks their formatting
    """
    # setup vars, create folder to save files
    user_folder = get_user_folder()
    study_folder = secure_filename(form.study_name.data)

    # -------------------------------------------------------------------------
    # save files
    files_dict = {}

    # new user? make folder
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    user_data_folder = os.path.join(user_folder, study_folder)
    if not os.path.exists(user_data_folder):
        os.makedirs(user_data_folder)

    # create dashboard folder if fs
    if bool(form.fs.data):
        dashboard_folder = os.path.join(user_folder, study_folder, 'dashboard')
        if not os.path.exists(dashboard_folder):
            os.makedirs(dashboard_folder)

    # save files with secure names
    for field, f in files.items():
        filename = secure_filename(f.filename)
        path = os.path.join(user_data_folder, filename)
        files_dict[field] = filename
        f.save(path)
        f.close()

    # -------------------------------------------------------------------------
    # check uploaded files
    # this import must be here, because get_user_folder is needed by check_files
    from backend.utils.check_uploaded_files import check_files
    status, format_errors = check_files(user_data_folder, files_dict, form)
    if status:
        metadata_cols = format_errors
        save_study_to_db(files_dict, form, metadata_cols)
        return json.dumps(dict(status='OK'))
    else:
        return json.dumps(dict(status='errors', errors=format_errors))


def save_study_to_db(files_dict, form, metadata_cols):
    """
    Saves the paths of files and parameters to the db
    """
    if not bool(form.annotation.data):
        species = None
    else:
        species = form.species.data

    # if we have dashboard cols, safely join them
    if metadata_cols is not None:
        metadata_cols = '_|_'.join(metadata_cols)
    study = models.Studies(author=current_user,
                           species=species,
                           study_name=secure_filename(form.study_name.data),
                           annotation=bool(form.annotation.data),
                           fs=bool(form.fs.data),
                           # if the autocorr checkbox is ticked we have two
                           # datasets, so it's not autocorr, so we save false
                           autocorr=not bool(form.autocorr.data),
                           dataset1_type=form.dataset1_type.data,
                           metadata_cols=metadata_cols,
                           timestamp=datetime.datetime.utcnow())
    if form.autocorr.data:
        setattr(study, 'dataset2_type', form.dataset2_type.data)

    for field, f in files_dict.items():
        # save the checked versions of the annotation files
        if field in ['annotation1', 'annotation2']:
            filename, extension = os.path.splitext(f)
            f = filename + '_checked' + extension
        setattr(study, field, f)
    db.session.add(study)
    db.session.commit()

    # increase total number of studies as well
    current_user.num_studies = current_user.num_studies + 1
    db.session.add(current_user)
    db.session.commit()

# -----------------------------------------------------------------------------
# ANALYSIS - HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def save_analysis(form, study_id):
    """
    Creates analysis folder, calls save_analysis_to_db and write_params_file
    """
    # setup vars to write params file
    user_folder = get_user_folder()
    study = models.Studies.query.get(study_id)
    study_name = study.study_name
    analysis_name = secure_filename(form.analysis_name.data)
    study_folder = os.path.join(user_folder, study_name)
    analysis_folder = os.path.join(study_folder, analysis_name)
    output_folder = os.path.join(analysis_folder, 'output')
    output_img_folder = os.path.join(output_folder, 'img')
    vis_folder = os.path.join(analysis_folder, 'vis')
    vis_genomic_folder = os.path.join(analysis_folder, 'vis_genomic')
    img_folder = os.path.join(analysis_folder, 'img')

    # define dict with all folders, we save it to params later
    folders = {
        "user_folder": user_folder,
        "study_folder": study_folder,
        "analysis_folder": analysis_folder,
        "output_folder": output_folder,
        "output_img_folder": output_img_folder,
        "vis_folder": vis_folder,
        "vis_genomic_folder": vis_genomic_folder,
        "img_folder": img_folder
    }

    # create folder for analysis
    if not os.path.exists(analysis_folder):
          os.makedirs(analysis_folder)

    # make output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # make img folder in output folder
    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)

    # make vis folder
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)

    # if genomic study make vis_genomic folder
    annot = bool(study.annotation)
    if annot and not os.path.exists(vis_genomic_folder):
        os.makedirs(vis_genomic_folder)

    # make img folder
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # write params.csv file
    write_params_file(folders, form, study_id)

    # save analysis to db
    save_analysis_to_db(form, study_id)


def write_params_file(folders, form, study_id):
    """
    Writes the params.csv file which is essential for each analysis to run.
    """
    study = models.Studies.query.get(study_id)

    # create param file
    params_file = os.path.join(folders["analysis_folder"], 'params.csv')
    params = open(params_file, 'w')

    # write header
    params.write(",value\n")

    # write folders
    for k, v in folders.iteritems():
        params.write("%s,%s\n" % (k, v))

    # write studies
    study_fields = ['annotation', 'autocorr', 'dataset1', 'annotation1',
                    'dataset2', 'annotation2', 'metadata_file', 'species']
    for i, f in enumerate(study_fields):
        field = getattr(study, f)
        if field is not None:
            params.write(f + ',' + str(field) + '\n')

    # write params
    param_fields = ['fs', 'fs_method', 'fs_cols', 'discard_overlap','alpha_val',
                    'lambda_val', 'multi_corr_method', 'constrain_corr',
                    'constrain_dist']
    for p in param_fields:
        field = getattr(form, p).data
        if field is not None:
            if p == 'fs' and study.fs:
                field = not bool(field)
            params.write(p + ',' + str(field) + '\n')
    params.close()


def save_analysis_to_db(form, study_id):
    """
    Saves parameters of analysis and start it on the cluster
    """
    # get if user study has dashboard and combine this with analysis form data
    fs = bool(models.Studies.query.get(study_id).fs)
    form_fs = not bool(form.fs.data)
    if fs:
        fs = form_fs
    analysis = models.Analyses(author=current_user,
                               study=models.Studies.query.get(study_id),
                               analysis_name=secure_filename(form.analysis_name.data),
                               status=1,
                               fs=fs,
                               fs_method=form.fs_method.data,
                               fs_col=form.fs_cols.data,
                               multi_corr_method=form.multi_corr_method.data,
                               alpha_val=form.alpha_val.data,
                               lambda_val=form.lambda_val.data,
                               discard_overlap=bool(form.discard_overlap.data),
                               constrain_corr=bool(form.constrain_corr.data),
                               constrain_dist=form.constrain_dist.data,
                               timestamp_start=datetime.datetime.utcnow())
    db.session.add(analysis)
    db.session.commit()

    # increase total number of studies as well
    current_user.num_analyses = current_user.num_analyses + 1
    db.session.add(current_user)
    db.session.commit()

# -----------------------------------------------------------------------------
# PROFILE - HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_studies_array():
    """
    Returns an array of object, so we can display in a popover the files of
    each study on the profile page.
    """
    studies_array = []
    studies = current_user.studies.all()
    for study in studies:
        study_dict = {}

        # get basic info of study
        study_dict['id'] = study.id
        study_dict['study_name'] = study.study_name
        study_dict['fs'] = bool(study.fs)

        # get files of study
        params = []
        params_names = ['Genomic datasets', 'Species', 'Metadata']
        params_fields = ['annotation', 'species', 'fs']
        for i, f in enumerate(params_fields):
            field = getattr(study, f)
            if field is not None:
                params.append({'field':params_names[i], 'value':field})
        params.append({'field':'Multi-omics study', 'value': not study.autocorr})
        study_dict['params'] = params

        files = []
        files_names = ['Dataset 1', 'Annotation 1', 'Dataset 2', 'Annotation 2',
                       'Metadata']
        files_fields = ['dataset1', 'annotation1', 'dataset2', 'annotation2',
                        'metadata_file']
        for i, f in enumerate(files_fields):
            field = getattr(study, f)
            if field is not None:
                files.append({'field':files_names[i], 'value':field})
        study_dict['files'] = files

        studies_array.append(study_dict)
    return studies_array


def get_analyses_array():
    """
    Returns an array of object, so we can display in a popover the parameters
    of each analysis on the profile page.
    """
    analyses_array = []
    analyses = current_user.analyses.all()
    user_folder = app.config['USER_PREFIX'] + str(current_user.id)
    for analysis in analyses:
        analysis_dict = {}
        # get basic info of analysis
        analysis_dict['id'] = analysis.id
        analysis_dict['analysis_name'] = analysis.analysis_name
        study = models.Studies.query.get(analysis.study_id)
        study_name = study.study_name
        if bool(study.autocorr):
            analysis_dict['data_file'] = 'dataset1'
        else:
            analysis_dict['data_file'] = 'dataset1_2'
        analysis_dict['status'] = analysis.status

        # build path for results .zip file
        analysis_folder = os.path.join(user_folder, secure_filename(study_name),
                                       secure_filename(analysis.analysis_name))
        results = analysis.analysis_name + '.zip'
        analysis_dict['results'] = os.path.join(analysis_folder, results)

        # collect all params for the analysis
        params = []
        params.append({'field':'Study name', 'value': study_name})
        param_names = ['Perform feature selection', 'Feature selection method',
                       'Metadata variable for FS', 'Multiple test correction method',
                       'Alpha', 'Lambda threshold', 'Discard overlapping correlations',
                       'Constrain correlations', 'Constrain distance']
        param_fields = ['fs', 'fs_method', 'fs_col', 'multi_corr_method',
                        'alpha_val', 'lambda_val', 'discard_overlap',
                        'constrain_corr', 'constrain_dist']

        # depending on the type of study and analysis we ignore fields
        analysis_fs = getattr(analysis, 'fs')
        if not analysis_fs:
            del param_names[1:3]
            del param_fields[1:3]
        if analysis_fs and not bool(study.annotation):
            del param_names[6:9]
            del param_fields[6:9]
        if not analysis_fs and not bool(study.annotation):
            del param_names[4:7]
            del param_fields[4:7]

        # write out the remaining fields
        for i, p in enumerate(param_fields):
            field = getattr(analysis, p)
            if field is not None:
                params.append({'field':param_names[i], 'value':field})

        analysis_dict['params'] = params
        analyses_array.append(analysis_dict)
    return analyses_array

# -----------------------------------------------------------------------------
# GENERAL - HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def security_check(user_id, study_id, analysis=False):
    """
    For a couple of views, we need to make sure that the user is accessing
    content that is his/her and not others.
    """
    # check if user is trying to access a dataset that's not theirs
    user_ok = user_id == current_user.id

    # check if the study belongs to the user
    study_ok = False

    # if delete analysis was clicked, we need to check analyses and analysis_id
    if analysis:
        studies = current_user.analyses.all()
    else:
        studies = current_user.studies.all()
    for study in studies:
        if study.id == study_id:
            study_ok = True
    return user_ok * study_ok


def get_user_folder():
    """
    Returns the folder of the current user, based on the config params.
    """
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'],
                               app.config['USER_PREFIX'] +
                               str(current_user.id))
    return user_folder


def get_study_folder(study_id):
    """
    Returns a path to the current user's study
    """
    user_folder = get_user_folder()
    study = models.Studies.query.get(study_id)
    study_name = study.study_name
    study_folder = os.path.join(user_folder, study_name)
    return study_folder
