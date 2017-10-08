import requests
import json
from . import app, models, appdir
from os import listdir
from os.path import join, splitext
from collections import defaultdict
from flask import session
from flask_security.forms import RegisterForm
from flask_wtf import Form, RecaptchaField
from flask_wtf.file import FileField
from wtforms import StringField, DecimalField, IntegerField, BooleanField, \
                    SelectField, ValidationError
from wtforms.validators import InputRequired, Length, DataRequired, \
                               NumberRange, Email
from werkzeug.utils import secure_filename
from flask_security import current_user


# =============================================================================
#
# UPLOAD FORM
#
# =============================================================================

# -----------------------------------------------------------------------------
# UPLOAD VALIDATORS
# -----------------------------------------------------------------------------

# custom validator to check that each dataset has a unique name
class UniqueDatasetName(object):
    def __init__(self, message="You already have a dataset with this name."):
        self.message = message

    def __call__(self, form, field):
        studies = current_user.studies.all()
        if len(studies) > 0:
            for study in studies:
                if study.study_name == field.data:
                    raise ValidationError(self.message)


# validator which makes a field required if another field(s) is set to true
class RequiredIf(DataRequired):
    def __init__(self, other_field_names, *args, **kwargs):
        self.other_field_names = other_field_names
        super(RequiredIf, self).__init__(*args, **kwargs)

    def __call__(self, form, field):
        # if all required fields are true then continue
        bool_state = True
        for f in self.other_field_names:
            bool_state *= bool(form._fields.get(f).data)
        if bool(bool_state):
            super(RequiredIf, self).__call__(form, field)


# check the file extensions on the server side as well (handle required radio)
class ExtensionCheck(object):
    def __init__(self, exts, required=None, message=None):
        self.required = required
        self.exts = exts
        if not message:
            extString = ','.join(exts)
            message = ('File must have one of the following extensions: %s.' %
                       (extString))
        self.message = message
    def __call__(self, form, field):
        # if field has required field, only run check if that field is checked
        if self.required is not None:
            # get boolean value of the required radio box
            self.requiredBool = bool(form._fields.get(self.required).data)
        else:
            self.requiredBool = True

        # we send back the size of the file for checking after the '__' so we
        # need this hack to get the real extension.
        file = field.data.split('__')[0]
        filename, extension = splitext(file)
        if extension not in self.exts and self.requiredBool:
            raise ValidationError(self.message)


class BooleanRequired(object):
    def __init__(self, message="This box needs to be checked."):
        self.message = message

    def __call__(self, form, field):
        if field.data is None or field.data is False:
            raise ValidationError(self.message)

# validator for the name fields in the register form
name_validator = [InputRequired(), Length(min=3,max=50)]

# -----------------------------------------------------------------------------
# GET SPECIES LIST FOR SPECIES SELECT-FIELD IN UPLOAD FORM
# -----------------------------------------------------------------------------

# gets the all the current binning files we have for all speces
bin_files = listdir(join(app.config['BACKEND_FOLDER'],'bins'))
species_d = {}
for f in bin_files:
    if f.split('.')[-1] == 'txt' and f.split('__')[1] == 'bins.txt':
        species_name = f.split('__')[0].replace('_',' ')
        if f not in species_d:
            species_d[species_name] = f
# get alphabetic ordered array from species_d
species_a = sorted(species_d.keys())
# generate name value pairs for species SelectField
species_data = zip(species_a,species_a)


# -----------------------------------------------------------------------------
# FORM
# -----------------------------------------------------------------------------

class UploadForm(Form):
    file_sel = 'Please select a file.'
    study_name = StringField('Name of the study', [InputRequired(),
                              Length(min=3,max=25), UniqueDatasetName()])
    species = SelectField('Species', coerce=str, choices=species_data,
                          default='Human - hg38')
    annotation = BooleanField('Genomic dataset?')
    dataset1 = FileField('Dataset file 1', [DataRequired(file_sel),
                         ExtensionCheck(exts=['.txt','.csv'])])
    annotation1 = FileField('Annotation file 1', [RequiredIf(['annotation'],
                            message=file_sel)])
    dataset1_type = StringField('Type of dataset 1', [InputRequired(),
                                Length(min=3,max=25)])
    autocorr = BooleanField('More than one dataset')
    dataset2 = FileField('Dataset file 2', [RequiredIf(['autocorr'],
                         message=file_sel), ExtensionCheck(exts=['.txt','.csv'],
                         required='autocorr')])
    dataset2_type = StringField('Type of dataset 2', [RequiredIf(['autocorr'])])
    annotation2 = FileField('Annotation file 2', [RequiredIf(['annotation',
                            'autocorr'], message=file_sel)])
    fs = BooleanField('Feature selection')
    metadata_file = FileField('Metadata file', [RequiredIf(['fs'],
                              message=file_sel), ExtensionCheck(exts=['.txt',
                              '.csv'], required='fs')])
    tc = BooleanField('Terms and conditions', [BooleanRequired()])

    # hidden field to indicate from the client side when sending the AJAX POST
    # reuqest if we are validating or submitting
    check = BooleanField('')

    # custom validator function to make sure that all files are different
    def validate(self):
        result = Form.validate(self)
        # only run this when checking the form, not when we are submitting
        if self.check.data:
            max_size = app.config['MAX_CONTENT_LENGTH']
            seen = set()
            files = [self.dataset1, self.annotation1]
            # if autocorr is clicked we need all 4 files to be  diff
            if self.autocorr.data:
                files += [self.dataset2, self.annotation2]
            # if fs is clicked we need all 5 files to be  diff
            if self.fs.data:
                files += [self.metadata_file]
            for field in files:
                # since we don't submit a proper file object just the name of
                # the files we can't access the .filename, so we just use the
                # .data to check names
                # filename = secure_filename(field.data.filename)
                clean_name = field.data.split('/')[-1].split('\\')[-1]
                filename = secure_filename(clean_name)
                if filename != '':
                    file_size = int(field.data.split('__')[-1])
                    if file_size > max_size:
                        max_file_size_string = app.config['MAX_FILE_SIZE']
                        field.errors.append('Maximum file size is ' +
                                            max_file_size_string + '.')
                        result = False
                    if filename in seen:
                        field.errors.append('Please select different files for'
                                            ' each data field.')
                        result = False
                    else:
                        seen.add(filename)
            return result


# =============================================================================
#
# REGISTRATION FORM
#
# =============================================================================

def load_uni_address_data():
    global uni_data, uni_loaded
    # instead of loading it from online we'll use the offline version because
    # it's easier to add exceptions to it
    # json_address = ("https://raw.githubusercontent.com/Hipo/university-"
    #                 "domains-list/master/world_universities_and_domains.json")
    # response = requests.get(json_address)
    # if response.status_code == 200:
    #     uni_json = response.json()
    # # if we cannot find it only use the offline version
    # else:
    json_url = join(appdir , "static/uni", "uni.json")
    uni_json = json.load(open(json_url))

    for uni in uni_json:
        uni_data[uni["domain"].lower()].append(uni["country"])
        uni_data[uni["domain"].lower()].append(uni["name"])
    uni_loaded = True

# load university json
uni_data = defaultdict(list)
uni_loaded = False
load_uni_address_data()


# check if an email address is an academic one
class AcademicEmailAddress(object):
    def __init__(self, message="This doesn't seem like a valid academic email "
                               "address. If we've made a mistake, please "
                               "contact us."):
        self.message = message
        if not uni_loaded:
            load_uni_address_data()

    def __call__(self, form, field):
        user_uni_domain = field.data.split("@")[1]
        user_uni_domain_ok = False
        for k, v in uni_data.iteritems():
            if user_uni_domain.endswith(k):
                user_uni_domain_ok = True
                break
        if not user_uni_domain_ok:
            raise ValidationError(self.message)

email_validators = [InputRequired(),
                    Email("Please provide a valid email address."),
                    AcademicEmailAddress()]


# -----------------------------------------------------------------------------
# FORM
# -----------------------------------------------------------------------------

# adds first and last name to Flask-Security's basic User model
class ExtendedRegisterForm(RegisterForm):
    first_name = StringField('First Name', name_validator)
    last_name = StringField('Last Name', name_validator)
    email = StringField('Email address', email_validators)


# =============================================================================
#
# ANALYSIS FORM
#
# =============================================================================

# -----------------------------------------------------------------------------
# ANALYSIS VALIDATORS
# -----------------------------------------------------------------------------

# custom validator to check that each analysis within a dataset has unique name
class UniqueAnalysisName(object):
    def __init__(self, message="You already have an analysis with this name"
                               " for this dataset."):
        self.message = message

    def __call__(self, form, field):
        current_study_id = session['study_id']
        if current_study_id is not None:
            current_study = models.Studies.query.get(current_study_id)
            analyses = current_study.analyses.all()

            if len(analyses) > 0:
                for analysis in analyses:
                    if analysis.analysis_name == field.data:
                        raise ValidationError(self.message)


# -----------------------------------------------------------------------------
# DATA FOR FS METHOD, MULTI CORR METHOD SELECT-FIELD
# -----------------------------------------------------------------------------

fs_labels = ['L1', 'JMI', 'Boruta', 'FDR']
fs_data = zip(fs_labels, fs_labels)

multi_labels = ['Benjamini-Hochberg', 'Benjamini-Yekutieli', 'Bonferroni']
multi_ind = ['fdr_bh','fdr_by','bonferroni']
multi_data = zip(multi_ind, multi_labels)

number_ranges = {
    'alpha': NumberRange(min=.001, max=1,
                         message='Has to be between 0.01 and 1'),
    'lambda': NumberRange(min=.01, max=1,
                          message='Has to be between 0.001 and 1'),
    'constrain': NumberRange(min=0, max=1000,
                             message='Has to be between 1 and 1000')
}


# -----------------------------------------------------------------------------
# FORM
# -----------------------------------------------------------------------------

class AnalysisForm(Form):
    analysis_name = StringField('Name of the analysis', [InputRequired(),
                                Length(min=3,max=25), UniqueAnalysisName()])
    fs = BooleanField('Skip feature selection')
    fs_method = SelectField('Feautre selection method', coerce=str,
                            choices=fs_data, default='L1')
    fs_cols = SelectField('Metadata variable', coerce=str)
    multi_corr_method = SelectField('Method for correction', coerce=str,
                                    choices=multi_data, default='fdr_by')
    alpha_val = DecimalField('&alpha; for the multiple correction method',
                             [number_ranges['alpha']], default=0.05)
    lambda_val = DecimalField('&Lambda; threshold for StARS',
                             [number_ranges['lambda']], default=0.1)
    discard_overlap = BooleanField('Discard overlapping correlations')
    constrain_corr = BooleanField('Constrain maximum distance of correlations')
    constrain_dist = IntegerField('Maximum distance of correlations',
                                  [number_ranges['constrain']], default=0)
    check = BooleanField('')
