# stellar-classification
Classification of Stars, Galaxies and Quasars. Sloan Digital Sky Survey DR17

Random forest on imaging-only features (4 colors + r) reaches a test macro F1 of 0.852 (baseline: 0.25). See the [analysis report](REPORT.md) for the full write-up.

## Live demo

**https://7ld2mronnk2hmj5mfmdup3h4rq0lxwbj.lambda-url.us-east-2.on.aws/**

- [`/`](https://7ld2mronnk2hmj5mfmdup3h4rq0lxwbj.lambda-url.us-east-2.on.aws/): enter five magnitudes, or try a star, galaxy or quasar example
- [`/docs`](https://7ld2mronnk2hmj5mfmdup3h4rq0lxwbj.lambda-url.us-east-2.on.aws/docs): interactive API documentation for `POST /predict`
- [`/health`](https://7ld2mronnk2hmj5mfmdup3h4rq0lxwbj.lambda-url.us-east-2.on.aws/health): status and the model version being served

Runs on AWS Lambda; the first request after a quiet period takes a few seconds while the function starts. See [deploy/README.md](deploy/README.md) for how it is deployed.
