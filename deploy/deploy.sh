#!/usr/bin/env bash
# Deploy from this laptop: smoke test → train + deploy gate → build → push → update the Lambda → check.
#
# Signed in as you (`aws login --profile stellar`): this AWS account type blocks GitHub's keyless
# sign-in (OIDC), so the pipeline runs here instead of in GitHub Actions. The steps and checks are
# the ones a CI deploy would run. Any failing step stops the script, so a model that fails the gate
# is never built or deployed, and the live site keeps serving the old one.
#
# Run from the project root, in the stellar conda env:  ./deploy/deploy.sh
set -euo pipefail

export AWS_PROFILE=stellar AWS_REGION=us-east-2
REPOSITORY=stellar-classifier-ecr
FUNCTION=myStellarClassificationFunction

echo '== 0. AWS sign-in'
if ! ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null); then
  echo 'Not signed in to AWS. Run: aws login --profile stellar' >&2
  exit 1
fi
REGISTRY=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo '== 1. Smoke test on the real data (must run, not skip)'
REQUIRE_SMOKE_TEST=1 python -m pytest -rs tests/test_smoke.py

echo '== 2. Train, run the deploy gate, store the model in S3'
python train.py --deploy
VERSION=$(python -c "import json; print(json.load(open('artifacts/metadata.json'))['model_version'])")
IMAGE=$REGISTRY/$REPOSITORY:$VERSION

echo "== 3. Build and push $IMAGE"
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
docker build --platform linux/arm64 -t "$IMAGE" .
docker push "$IMAGE"

echo '== 4. Point the Lambda at the new image'
aws lambda update-function-code --function-name "$FUNCTION" --image-uri "$IMAGE" > /dev/null
aws lambda wait function-updated-v2 --function-name "$FUNCTION"

echo '== 5. Check /health serves the new model'
LIVE_URL=$(python -c "from src.config import LIVE_URL; print(LIVE_URL)")
# The first start after a new image can exceed Lambda's 10 s init limit and time out,
# so retry for up to ~2 minutes before calling the deploy failed
for attempt in $(seq 1 12); do
  served=$(curl -s --max-time 30 "$LIVE_URL/health" \
    | python -c "import json, sys; print(json.load(sys.stdin).get('model_version', ''))" 2>/dev/null || true)
  echo "attempt $attempt: serving '$served', expected '$VERSION'"
  if [ "$served" = "$VERSION" ]; then
    echo "Deployed $VERSION"
    exit 0
  fi
  sleep 10
done
echo "/health does not report $VERSION. Roll back." >&2
exit 1
