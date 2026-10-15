# Deployment

The API runs on **AWS Lambda** as a container image, behind a public **Lambda Function URL**.

```
docker build ──▶ Amazon ECR ──▶ AWS Lambda (arm64, FastAPI via Mangum) ──▶ Function URL (public HTTPS)
                                     │
                                     └──▶ CloudWatch Logs
```

Live: https://7ld2mronnk2hmj5mfmdup3h4rq0lxwbj.lambda-url.us-east-2.on.aws/

## AWS resources

| Resource | Name | Settings |
|---|---|---|
| ECR repository | `stellar-classifier-ecr` | Private, AES-256 encryption; images tagged with the model version |
| IAM role | `stellar-classifier-lambda` | Trusted by `lambda.amazonaws.com` only; one policy, `AWSLambdaBasicExecutionRole` (write logs) |
| Lambda function | `myStellarClassificationFunction` | Event function, container image, **arm64**, 1024 MB memory, 30 s timeout, reserved concurrency 2 |
| Function URL | (see above) | Auth type `NONE`; resource policy allows invoking only through the URL |
| Log group | `/aws/lambda/myStellarClassificationFunction` | Created by Lambda on first run |

Region: `us-east-2`. The AWS CLI profile is `stellar` (`aws login --profile stellar`).

## First-time setup

Commands assume the project root as the working directory. Steps 3–5 were done in the AWS console; the CLI equivalent is shown for each.

**0. Shortcuts for the terminal session**
```bash
export AWS_PROFILE=stellar AWS_REGION=us-east-2
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPO_URI=$ACCOUNT_ID.dkr.ecr.us-east-2.amazonaws.com/stellar-classifier-ecr
export FUNCTION=myStellarClassificationFunction
export TAG=$(python -c "import json; print(json.load(open('artifacts/metadata.json'))['model_version'])")
```

**1. Train and build** (the image must be arm64 to match the function)
```bash
python train.py
docker build --platform linux/arm64 -t stellarclassification:latest .
```

**2. Create the ECR repository** (console: ECR → Create repository)
```bash
aws ecr create-repository --repository-name stellar-classifier-ecr
```

**3. Push the image**
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-east-2.amazonaws.com
docker tag stellarclassification:latest $REPO_URI:$TAG
docker push $REPO_URI:$TAG
```

**4. Create the execution role** (console: IAM → Roles → Create role → AWS service: Lambda)
```bash
aws iam create-role --role-name stellar-classifier-lambda --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name stellar-classifier-lambda --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

**5. Create the function** (console: Lambda → Create function → Event function → Container image)
```bash
aws lambda create-function --function-name $FUNCTION --package-type Image \
  --code ImageUri=$REPO_URI:$TAG \
  --role arn:aws:iam::$ACCOUNT_ID:role/stellar-classifier-lambda \
  --architectures arm64 --memory-size 1024 --timeout 30
aws lambda wait function-active-v2 --function-name $FUNCTION
```
The console's defaults need changing afterwards: memory 128 MB → 1024 MB, timeout 3 s → 30 s, and the
default execution role (`PowerUserRole`, near-full account access) → `stellar-classifier-lambda`:
```bash
aws lambda update-function-configuration --function-name $FUNCTION --memory-size 1024 --timeout 30 \
  --role arn:aws:iam::$ACCOUNT_ID:role/stellar-classifier-lambda
```
Cap how many copies of the function can run at once. The URL is public, so this limits how fast a
flood of requests can spend money; requests beyond the cap get HTTP 429 and cost nothing. Two is
plenty for a demo (a warm `/predict` takes ~20 ms):
```bash
aws lambda put-function-concurrency --function-name $FUNCTION --reserved-concurrent-executions 2
```

**6. Create the public Function URL** (console: function → Configuration → Function URL → Create, auth type `NONE`)

The console also adds the two resource-policy statements below. With the CLI, add them yourself; the
condition on the second keeps direct invocation private.
```bash
aws lambda create-function-url-config --function-name $FUNCTION --auth-type NONE
aws lambda add-permission --function-name $FUNCTION --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE
aws lambda add-permission --function-name $FUNCTION --statement-id FunctionURLAllowInvokeAction \
  --action lambda:InvokeFunction --principal "*" --invoked-via-function-url
```

## Deploying a new version

After retraining or changing serving code: run steps 0, 1 and 3 (new `TAG`), then point the function at the new image.
```bash
aws lambda update-function-code --function-name $FUNCTION --image-uri $REPO_URI:$TAG
aws lambda wait function-updated-v2 --function-name $FUNCTION
```
Check that `/health` reports the new `model_version`, and open the URL once before sharing it (see cold starts below).

**Rollback:** run `update-function-code` with the previous image tag. Every pushed image stays in ECR under its model version.

## Testing

**Before deploying**, run the image locally with the Lambda emulator built into the base image:
```bash
docker run --rm -p 9000:8080 stellarclassification:latest
curl -s -X POST localhost:9000/2015-03-31/functions/function/invocations -d @deploy/events/health.json
```
`deploy/events/` holds Function URL requests for `GET /health`, `GET /`, `POST /predict` and an invalid
`POST /predict` (expected `statusCode` 200, 200, 200 and 422).

**After deploying**:
```bash
URL=https://7ld2mronnk2hmj5mfmdup3h4rq0lxwbj.lambda-url.us-east-2.on.aws
curl -s $URL/health
curl -s -X POST $URL/predict -H 'Content-Type: application/json' \
  -d '{"objects": [{"u": 21.348, "g": 21.264, "r": 20.892, "i": 20.939, "z": 20.794}]}'
```

## Performance and cold starts

Measured 2026-10-15 (1024 MB, arm64, 915 MB image):

| | Time |
|---|---|
| Cold start (`Init Duration`) | ~2.2 s |
| Request inside Lambda, warm | ~3 ms (`/health`), ~20 ms (`/predict`) |
| Memory used | ~360 MB of 1024 MB |

- **The first start after a new image is deployed can take more than 10 s** while Lambda caches the image.
  Lambda's init phase is limited to 10 s, so that first request can time out. Open the URL once after
  every deploy, before sharing it.
- For container images, AWS bills the init time as well as the request time.

To force a cold start for measurement, change any configuration value (e.g. an environment variable),
send one request, then read `Init Duration` from the `REPORT` lines:
```bash
aws logs tail /aws/lambda/$FUNCTION --since 30m | grep REPORT
```
