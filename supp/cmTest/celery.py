from __future__ import absolute_import
from celery import Celery

app = Celery('cmTest',
             broker='amqp://test:test123@localhost/test_vhost',
             # backend='rpc://',
             include=['cmTest.tasks'])
