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
| S3 bucket | `amzn-s3-stellar-data` | Private (all four Block Public Access settings on), versioning on, AES-256 encryption; training data at `raw/star_classification.csv`; every model that passed the deploy gate at `models/<model_version>/` |
| ECR repository | `stellar-classifier-ecr` | Private, AES-256 encryption; images tagged with the model version; tags immutable; lifecycle policy keeps the 3 newest images |
| IAM role | `stellar-classifier-lambda` | Trusted by `lambda.amazonaws.com` only; one policy, `AWSLambdaBasicExecutionRole` (write logs) |
| Lambda function | `myStellarClassificationFunction` | Event function, container image, **arm64**, 1024 MB memory, 30 s timeout, reserved concurrency 2 |
| Function URL | (see above) | Auth type `NONE`; resource policy allows invoking only through the URL |
| Log group | `/aws/lambda/myStellarClassificationFunction` | Created by Lambda on first run |

Region: `us-east-2`. The AWS CLI profile is `stellar` (`aws login --profile stellar`).

## Training data (S3)

`train.py` reads the training data from S3. The Lambda never does: the model is baked into the image,
so the function's role has no S3 permission.

Versioning keeps every upload: each one gets a new `VersionId`, and older versions stay readable. That
`VersionId` identifies the exact data a model was trained on.

**Create the bucket** (console: S3 → Create bucket; Block Public Access and AES-256 encryption are on by default)
```bash
aws s3api create-bucket --bucket amzn-s3-stellar-data \
  --create-bucket-configuration LocationConstraint=us-east-2
aws s3api put-bucket-versioning --bucket amzn-s3-stellar-data \
  --versioning-configuration Status=Enabled
```

**Upload the data** (the bucket and key are `DATA_BUCKET` and `DATA_KEY` in `src/config.py`)
```bash
aws s3 cp data/star_classification.csv s3://amzn-s3-stellar-data/raw/star_classification.csv
```

**List the stored versions**
```bash
aws s3api list-object-versions --bucket amzn-s3-stellar-data --prefix raw/star_classification.csv \
  --query 'Versions[].{VersionId:VersionId,Latest:IsLatest,Modified:LastModified}' --output table
```

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

## Deploying (`deploy/deploy.sh`)

Deploys run from this laptop, signed in as you:
```bash
aws login --profile stellar
./deploy/deploy.sh
```

```
smoke test on the real data ─▶ train + deploy gate ─▶ store model in S3 ─▶ build + push image
                            ─▶ update the Lambda ─▶ check /health serves the new version
```

- **Why not GitHub Actions:** CD from GitHub needs GitHub to sign in to AWS. The keyless way (OIDC) is
  blocked in this account: the AWS-managed organization behind the new sign-up experience denies every
  `iam:*OpenIDConnectProvider*` action in a service control policy, which no permission inside the account
  can override. The alternative, long-lived access keys stored in GitHub, was not used. GitHub Actions
  still runs the tests on every push (`.github/workflows/ci.yml`).
- **Deploy gate** (`src/gate.py`, run by `train.py --deploy`): the live model (found through `/health`,
  downloaded from `models/<version>/`) and the new model are scored on the same new test split. The new
  model deploys if its macro F1 is at least 0.84 and at least the live model's minus 0.005. If the live
  model can't be compared, only the 0.84 floor applies and the output says why. The result is stored in
  `models/<version>/metrics.json`.
- **A failing step stops the script.** A model that fails the gate is never built or deployed, and the
  live site keeps serving the old one.
- Commit before deploying, so the code behind the live image is in git.

### One-time setup

Run these once, signed in (`aws login --profile stellar`), from the project root.

**1. ECR: immutable tags, keep the 3 newest images**
```bash
aws ecr put-image-tag-mutability --repository-name stellar-classifier-ecr --image-tag-mutability IMMUTABLE \
  --profile stellar --region us-east-2
aws ecr put-lifecycle-policy --repository-name stellar-classifier-ecr \
  --lifecycle-policy-text file://deploy/ecr-lifecycle.json --profile stellar --region us-east-2
```
With 3 images kept, rollback reaches two versions back. Older models stay in `models/` in S3 and can
be rebuilt into an image.

**2. Put the live model into `models/`**, so the first deploy has a champion to compare against.
`artifacts/` must hold the model that is live: check the two versions match first.
```bash
curl -s https://7ld2mronnk2hmj5mfmdup3h4rq0lxwbj.lambda-url.us-east-2.on.aws/health
export LIVE=$(python -c "import json; print(json.load(open('artifacts/metadata.json'))['model_version'])")
echo $LIVE
aws s3 cp artifacts/model.joblib s3://amzn-s3-stellar-data/models/$LIVE/model.joblib --profile stellar
aws s3 cp artifacts/metadata.json s3://amzn-s3-stellar-data/models/$LIVE/metadata.json --profile stellar
```

## Deploying step by step (without the gate) and rollback

`deploy/deploy.sh` is the normal way. Step by step instead: run steps 0, 1 and 3 of "First-time setup"
(new `TAG`), then point the function at the new image.
```bash
aws lambda update-function-code --function-name $FUNCTION --image-uri $REPO_URI:$TAG
aws lambda wait function-updated-v2 --function-name $FUNCTION
```
Check that `/health` reports the new `model_version`, and open the URL once before sharing it (see cold starts below).

**Rollback:** run `update-function-code` with the previous image tag (the 3 newest images are kept;
`aws ecr list-images --repository-name stellar-classifier-ecr` lists them). If the deploy script's
`/health` check fails, roll back this way.

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
