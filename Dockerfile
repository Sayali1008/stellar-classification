# Lambda container image for the stellar classifier API (FastAPI + Mangum).
#
# Build for arm64 (Apple Silicon; the Lambda function must also be created as arm64):
#   docker build --platform linux/arm64 -t stellar-classifier .
# Run locally with the Lambda emulator built into the base image, then send it a request:
#   docker run --rm -p 9000:8080 stellar-classifier
#   curl -s -X POST localhost:9000/2015-03-31/functions/function/invocations -d @deploy/events/health.json

# AWS's official Lambda base image: Python 3.12 + the Lambda runtime + a local emulator
FROM public.ecr.aws/lambda/python:3.12

# Packages first: this slow layer is reused when only the code or the model changes
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Only what serving needs: no training script, notebook or data
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY artifacts/ ${LAMBDA_TASK_ROOT}/artifacts/

# Lambda calls the `handler` object in app/main.py
CMD ["app.main.handler"]
