# This is inspired from https://docs.celeryq.dev/en/stable/getting-started/introduction.html
# Download Celery and Redis (broker)
from celery import Celery

celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',  # Redis as broker
    backend= 'mongodb://username:password@host:port/dbname' # MongoDB as backend
)

celery_app.conf.task_serializer = 'json'
celery_app.conf.result_serializer = 'json'
celery_app.conf.accept_content = ['json']
