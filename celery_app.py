# This is inspired from https://docs.celeryq.dev/en/stable/getting-started/introduction.html
# Download Celery and Redis (broker)
from celery import Celery

celery_app = Celery(
    'fastapi_turicreate_images',
    broker='redis://localhost:6379/0',  # Redis as broker
    backend= 'mongodb+srv://omarcastelan:seFEZm1yn2EsKGyZ@smu8392coylef2024.l1ff5.mongodb.net/?retryWrites=true&w=majority&appName=SMU8392CoyleF2024' # MongoDB as backend
)

celery_app.conf.task_serializer = 'json'
celery_app.conf.result_serializer = 'json'
celery_app.conf.accept_content = ['json']