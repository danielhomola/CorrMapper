# coding=utf-8
import os

appdir = os.path.abspath(os.path.dirname(__file__))
ROOTDIR = os.path.abspath(os.path.join(appdir, os.pardir))

# -----------------------------------------------------------------------------
# Flask settings
# http://flask.pocoo.org/docs/config/#configuring-from-files
# -----------------------------------------------------------------------------

SECRET_KEY = "SOMETHING_LONG_AND_COMPLICATED"
FLASH_MESSAGES = True
DEBUG = False
TESTING = False

# -----------------------------------------------------------------------------
# CorrMapper settings
# -----------------------------------------------------------------------------

#  name of the app
APP_NAME = 'yourdomain'
# backend files
BACKEND_FOLDER = os.path.join(ROOTDIR,'backend')
# what to use before user id as a string
USER_PREFIX = 'user'
# what to use before study id as a string
DATASET_PREFIX = 'study'
# where to upload study
UPLOAD_FOLDER = os.path.join(ROOTDIR, 'userData')
# where to copy failed analysis so we can debug them later
FAILED_FOLDER = os.path.join(ROOTDIR, 'failedAnalyses')

# max file size in upload 100 MB
MAX_CONTENT_LENGTH = 100 * 1024 * 1024
# max file size in upload as string
MAX_FILE_SIZE = '100 MB'
# maximum number of studies allowed per user at any time-point
ACTIVE_STUDY_PER_USER = 3
# maximum number of studies allowed per user altogether (delete > re-upload)
STUDY_PER_USER = 5
# maximum number of analysis allowed per user at any time-point
ACTIVE_ANALYSIS_PER_USER = 6
# maximum number of analysis allowed per user altogether (delete > re-upload)
ANALYSIS_PER_USER = 10

# the min number of samples that we need in both datasets and dashboard file
INTERSECTING_SAMPLES = 15
# number of maximum samples we support per dataset
MAX_SAMPLES = 500
# number of maximum features we support per dataset
MAX_FEATURES = 25000
# minimum number of numeric features in a dataset
MINIMUM_FEATURES = 10
# number of dashboard columns we support per analysis
MAX_METADATA_COLUMNS = 15
# minimum number of samples per class
MIN_SAMPLE_PER_CLASS = 15
# number of PCA components to show in metadata explorer
NUM_PCA_COMPONENTS = 5
# -----------------------------------------------------------------------------
# Flask-SQLAlchemy
# http://pythonhosted.org/Flask-SQLAlchemy/config.html
# -----------------------------------------------------------------------------

SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(ROOTDIR, "yourdomain.db")
SQLALCHEMY_TRACK_MODIFICATIONS = True

# -----------------------------------------------------------------------------
# Flask-Security
# http://pythonhosted.org/Flask-Security/configuration.html
# -----------------------------------------------------------------------------

SECURITY_EMAIL_SENDER = "no-reply@yourdomain.com"

SECURITY_POST_LOGIN_VIEW = "profile"
SECURITY_POST_REGISTER_VIEW = "profile"
SECURITY_POST_CONFIRM_VIEW = "profile"
SECURITY_POST_CHANGE_VIEW = "index"
SECURITY_POST_RESET_VIEW = "index"
SECURITY_POST_LOGOUT_VIEW = "index"

SECURITY_CONFIRMABLE = True
SECURITY_REGISTERABLE = True
SECURITY_RECOVERABLE = True
SECURITY_CHANGEABLE = True
SECURITY_TRACKABLE = True

SECURITY_PASSWORD_HASH = "bcrypt"
SECURITY_PASSWORD_SALT = SECRET_KEY
SECURITY_CONFIRM_SALT = SECRET_KEY
SECURITY_RESET_SALT = SECRET_KEY
SECURITY_LOGIN_SALT = SECRET_KEY
SECURITY_REMEMBER_SALT = SECRET_KEY
SECURITY_DEFAULT_REMEMBER_ME = True

SECURITY_MSG_INVALID_PASSWORD = ("Bad username or password", "error")
SECURITY_MSG_PASSWORD_NOT_PROVIDED = ("Bad username or password", "error")
SECURITY_MSG_USER_DOES_NOT_EXIST = ("Bad username or password", "error")
SECURITY_MSG_PASSWORD_NOT_SET = ("No password is set for this user. Sign in "
                                 "with your registered provider on the right.",
                                 "error")

# -----------------------------------------------------------------------------
# FLASK ADMIN
# -----------------------------------------------------------------------------

ADMIN_NAME = "admin"
ADMIN_EMAIL = "admin@admin.com"
ADMIN_PASSWORD = "admin_pass"

# -----------------------------------------------------------------------------
# Flask-Babel
# http://pythonhosted.org/Flask-Babel/
# -----------------------------------------------------------------------------

BABEL_DEFAULT_LOCALE = "en"
BABEL_DEFAULT_TIMEZONE = "UTC"

# -----------------------------------------------------------------------------
# Flask-Mail
# http://pythonhosted.org/Flask-Mail/
#
# if TLS = True then port needs to be 587, if SSL = True it needs to 465. With
# 465 logging SMTPhandler doesn't work. So TSL it is.
# -----------------------------------------------------------------------------

MAIL_SERVER = 'AWS_MAIL_SERVER'
MAIL_PORT = 587
MAIL_USE_TLS = True
MAIL_USE_SSL = False
MAIL_USERNAME = 'AWS_SES_KEY'
MAIL_PASSWORD = 'AWS_SES_PASS'
#MAIL_DEBUG = True
# email addresses to send errors and logs
MAIL_DEFAULT_SENDER = "no-reply@corrmapper.com"
ADMINS = ['admin1@yourdomain.com', 'admin2@yourdomain.com']

# -----------------------------------------------------------------------------
# Celery
# http://blog.miguelgrinberg.com/post/using-celery-with-flask
# -----------------------------------------------------------------------------

CELERY_BROKER_URL = 'amqp://'
CELERY_SEND_TASK_ERROR_EMAILS = True
# we terminate after 60 minutes
CELERY_TASK_SERIALIZER = 'json'
CELERYD_TASK_TIME_LIMIT = 3600
SERVER_EMAIL = "admin@yourdomain.com"
EMAIL_HOST = MAIL_SERVER
EMAIL_HOST_USER = MAIL_USERNAME
EMAIL_HOST_PASSWORD = MAIL_PASSWORD
EMAIL_PORT = MAIL_PORT
EMAIL_USE_SSL = MAIL_USE_SSL
EMAIL_USE_TLS = MAIL_USE_TLS
