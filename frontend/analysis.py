import datetime
import os
import zipfile
import shutil
import traceback
from celery.exceptions import Terminated
from flask_mail import Message
from backend.bins import binning
from backend.corr import corr
from backend.fs import fs
from backend.genomic import genomic
from backend.utils import io_params
from frontend import app, db, models, create_celery_app, mail

celery = create_celery_app()
# have to set ADMIN config param here, because it would clash with Flask-Mail's
celery.conf.ADMINS = [('CorrMapper', 'corrmapper@corrmapper.com')]

@celery.task(throws=(Terminated, ), name='frontend.analysis.run_analysis')
def run_analysis(current_user_id):
    with celery.app.app_context():
        # ----------------------------------------------------------------------
        # LOAD STUDY AND ANALYSIS VARIABLES
        # ----------------------------------------------------------------------

        user = models.User.query.get(current_user_id)
        uid = current_user_id
        analysis = user.analyses.filter_by(status=1).first()
        analysis_id = analysis.id
        study = models.Studies.query.get(analysis.study_id)

        # get folders of the analysis - this is needed for load_params
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'],
                                   app.config['USER_PREFIX'] +
                                   str(current_user_id))
        study_folder = os.path.join(user_folder, study.study_name)
        analysis_folder = os.path.join(study_folder, analysis.analysis_name)

        # load params file as a dict and define folders from it
        params = io_params.load_params(analysis_folder)
        output_folder = params['output_folder']
        failed_folder = app.config['FAILED_FOLDER']

        # ----------------------------------------------------------------------
        # BINNING OF GENOMIC ELEMENTS
        # ----------------------------------------------------------------------

        try:
            if params['annotation']:
                # first analysis and annotations haven't been binned?
                binned_annot = params['annotation1'].replace('_checked',
                                                             '_binned')
                if not os.path.exists(binned_annot):
                    bin_file = study.species.replace(' ', '_') + '__bins.txt'
                    path = os.path.join(app.config['BACKEND_FOLDER'], 'bins',
                                        bin_file)
                    params['bin_file'] = path
                    params = binning.bin_genomic_elements_main(params)
                    params['binned'] = True
        except:
            delete_analysis(uid, analysis_id, analysis_folder, failed_folder)
            send_fail_mail(user.email, user.first_name, analysis.analysis_name)
            app.logger.error('Binning genomic elements failed for analysis: %d'
                             '\n%s' % (analysis_id, traceback.format_exc()))
            return False

        # ----------------------------------------------------------------------
        # TOP VARIANCE FEATURES AND FEATURE SELECTION
        # ----------------------------------------------------------------------

        try:
            params = fs.fs_main(params)
            if not params['fs_done']:
                delete_analysis(uid, analysis_id, analysis_folder, failed_folder)
                subject = 'Your CorrMapper job could not be completed'
                message = \
                    ('Your analysis (named: %s) could not be completed '
                     'because the chosen feature selection method did not '
                     'find enough discriminatory features with respect to the'
                     ' %s metadata variable. \n\nWithout this we cannot '
                     'complete the subsequent graph estimation (Glasso) '
                     'procedure.\n\nPlease either choose a different feature '
                     'selection method, or use another metadata variable.'
                     '\nCorrMapper Team'
                     % (analysis.analysis_name, params['fs_cols']))
                send_mail(user.email, user.first_name, analysis.analysis_name,
                          subject, message)
                return False
        except:
            delete_analysis(uid, analysis_id, analysis_folder, failed_folder)
            send_fail_mail(user.email, user.first_name, analysis.analysis_name)
            app.logger.error('Feature selection failed for analysis: %d'
                             '\n%s' % (analysis_id, traceback.format_exc()))
            return False

        # ----------------------------------------------------------------------
        # CALCULATE CORR NETWORK, P-VALUES, WRITE JS VARS AND DATA FOR VIS
        # ----------------------------------------------------------------------

        try:
            params = corr.corr_main(params)
            if not params['corr_done']:
                delete_analysis(uid, analysis_id, analysis_folder, failed_folder)
                subject = 'Your CorrMapper job could not be completed'
                message = \
                    ('Your analysis (named: %s) could not be completed '
                     'because one of the correlation matrices returned by the'
                     'GLASSO algorithm is empty. This could happen if you have'
                     'very few samples, select overly harsh p-value cut-off, or'
                     'the selected feature selection algorithm did not find'
                     'enough relevant features.\n\n You can try to run '
                     'CorrMapper with a different metadata variable, feature '
                     'selection method, or dataset.'
                     '\nCorrMapper Team'
                     % analysis.analysis_name)
                send_mail(user.email, user.first_name, analysis.analysis_name,
                          subject, message)
                return False
        except:
            delete_analysis(uid, analysis_id, analysis_folder, failed_folder)
            send_fail_mail(user.email, user.first_name, analysis.analysis_name)
            app.logger.error('Correlation calculation failed for analysis: %d'
                             '\n%s' % (analysis_id, traceback.format_exc()))
            return False

        # ----------------------------------------------------------------------
        # VIS_GENOMIC
        # ----------------------------------------------------------------------

        try:
            if params['annotation']:
                params = genomic.genomic_main(params)
        except:
            delete_analysis(uid, analysis_id, analysis_folder, failed_folder)
            send_fail_mail(user.email, user.first_name, analysis.analysis_name)
            app.logger.error('Writing vis_genomic failed for analysis: %d'
                             '\n%s' % (analysis_id, traceback.format_exc()))
            return False

        # ----------------------------------------------------------------------
        # ZIP RESULTS FOLDER, DELETE OUTPUT FOLDER
        # ----------------------------------------------------------------------

        try:
            zip_study(analysis.analysis_name, analysis_folder, output_folder)
        except:
            delete_analysis(uid, analysis_id, analysis_folder, failed_folder)
            send_fail_mail(user.email, user.first_name, analysis.analysis_name)
            app.logger.error('Making zip from output failed for analysis: %d'
                             '\n%s' % (analysis_id, traceback.format_exc()))
            return False

        # ----------------------------------------------------------------------
        # SAVE PARAMS, SEND EMAIL
        # ----------------------------------------------------------------------

        try:
            io_params.write_params(analysis_folder, params)
            send_mail(user.email, user.first_name, analysis.analysis_name)
            analysis.status = 2
            analysis.timestamp_finish = datetime.datetime.utcnow()
            db.session.commit()
            return True
        except:
            delete_analysis(uid, analysis_id, analysis_folder, failed_folder)
            send_fail_mail(user.email, user.first_name, analysis.analysis_name)
            app.logger.error('End of pipeline failed for analysis: %d'
                             '\n%s' % (analysis_id, traceback.format_exc()))
            return False


def send_fail_mail(email, first_name, analysis_name):
    subject = 'Your CorrMapper job could not be completed'
    message = ('Your analysis (named: %s) encountered a bug in our system and'
               ' could not be completed. We received a report of this and will'
               ' get back to you once we fixed the issue. Until then, you can'
               ' try to run CorrMapper with a different metadata variable,'
               ' feature selection method, or dataset.\n\nThank you for your'
               ' understanding!\n\nCorrMapper Team' % analysis_name)
    send_mail(email, first_name, analysis_name, subject, message)


def send_mail(email, first_name, analysis_name, subject=None, message=None):
    # default success message
    if message == None:
        subject = 'Your CorrMapper job finished'
        message = ('Your analysis (named: %s) has finished running. '
                  'You can check the results in your profile under the Analyses'
                  ' tab. \nCorrMapper Team' % analysis_name)

    msg = Message(subject, recipients=[email])
    if first_name is None:
        hi = 'Hi,\n\n'
    else:
        hi = 'Hi %s, \n\n' % first_name
    msg.body = (hi + message)
    mail.send(msg)


def zip_study(analysis_name, analysis_folder, output_folder):
    """
    Writes a .zip file from the output folder and deletes the output_folder
    """
    zip_path = os.path.join(analysis_folder, analysis_name + '.zip')
    zipf = zipfile.ZipFile(zip_path , 'w', zipfile.ZIP_DEFLATED)
    of_len = len(output_folder) + 1
    for root, dirs, files in os.walk(output_folder):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, file_path[of_len:])
    zipf.close()

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


def terminate_analysis(task_id):
    """
    Kills running analysis if the users deletes it from profile
    """
    try:
        celery.control.revoke(task_id, terminate=True)
    except Terminated:
        pass


def get_failed_name(user, analysis_id, counter):
    return ('failed_u_' + str(user) + '_a_' + str(analysis_id) +
            '_' + str(counter))


def delete_analysis(user, analysis_id, analysis_folder, failed_folder):
    # delete from database, get study folder
    analysis = models.Analyses.query.get(analysis_id)
    db.session.delete(analysis)
    db.session.commit()
    # move analysis folder to failed folder so we can debug it later
    counter = 1

    failed_name = get_failed_name(user, analysis_id, counter)
    failed_path = os.path.join(failed_folder, failed_name)
    # if a failed analysis with this id already exist try again till we are ok
    if os.path.exists(failed_path):
        counter += 1
        failed_name = get_failed_name(user, analysis_id, counter)
        failed_path = os.path.join(failed_folder, failed_name)
        while os.path.exists(failed_path):
            counter += 1
            failed_name = get_failed_name(user, analysis_id, counter)
            failed_path = os.path.join(failed_folder, failed_name)
    shutil.copytree(analysis_folder, failed_path)
    # delete analysis folder from user folder
    if os.path.exists(analysis_folder):
        shutil.rmtree(analysis_folder)
