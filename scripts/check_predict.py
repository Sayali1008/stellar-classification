"""Check the inference module (src/schemas.py + src/predict.py) against the saved model.

Prints PASS/FAIL for each check and exits with code 1 if any check fails.
Needs artifacts/ from `python train.py`; does not need the training data.

Run from the project root:  python -m scripts.check_predict
"""
import subprocess
import sys
import time

import sklearn
from pydantic import ValidationError

from src.predict import load_model, model_version, predict
from src.schemas import StellarObject

# Real test-set objects the model classifies confidently
EXAMPLES = {
    'STAR': dict(u=21.481, g=20.110, r=19.479, i=19.276, z=19.240),
    'GALAXY': dict(u=22.345, g=20.897, r=19.409, i=18.837, z=18.491),
    'QSO': dict(u=21.348, g=21.264, r=20.892, i=20.939, z=20.794),
}
VALID = EXAMPLES['STAR']

failures = []


def check(name, passed, detail=''):
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f'  ({detail})' if detail else ''))
    if not passed:
        failures.append(name)


def rejects(fields):
    """True if StellarObject refuses these fields with a ValidationError."""
    try:
        StellarObject(**fields)
    except ValidationError:
        return True
    return False


def check_examples():
    print('\n-- Predictions on known objects')
    print(f'   model version: {model_version()}')
    preds = predict([StellarObject(**fields) for fields in EXAMPLES.values()])
    for true_class, pred in zip(EXAMPLES, preds):
        top = pred.probabilities[pred.predicted_class]
        check(f'{true_class} predicted as {true_class}', pred.predicted_class == true_class,
              f'got {pred.predicted_class} {pred.probabilities}')
        check(f'{true_class} label carries the highest probability', top == max(pred.probabilities.values()),
              f'{pred.predicted_class}={top}')
        check(f'{true_class} probabilities sum to ~1', abs(sum(pred.probabilities.values()) - 1) < 0.01)
        check(f'{true_class} has one probability per class',
              set(pred.probabilities) == {'GALAXY', 'QSO', 'STAR'}, str(sorted(pred.probabilities)))


def check_order_preserved():
    print('\n-- Batch order')
    order = ['QSO', 'STAR', 'GALAXY', 'STAR']
    preds = predict([StellarObject(**EXAMPLES[c]) for c in order])
    check('results come back in input order', [p.predicted_class for p in preds] == order,
          str([p.predicted_class for p in preds]))


def check_validation():
    print('\n-- Input validation')
    check('rejects -9999 placeholder', rejects({**VALID, 'u': -9999}))
    check('rejects NaN', rejects({**VALID, 'u': float('nan')}))
    check('rejects infinity', rejects({**VALID, 'u': float('inf')}))
    check('rejects non-numeric text', rejects({**VALID, 'u': 'abc'}))
    check('rejects unknown field', rejects({**VALID, 'R': 19.5}))
    check('rejects missing field', rejects({k: v for k, v in VALID.items() if k != 'z'}))
    check('accepts numeric text like "21.5"', not rejects({**VALID, 'u': '21.5'}))


def check_edge_cases():
    print('\n-- Edge cases')
    check('empty batch returns []', predict([]) == [])
    batch = [StellarObject(**VALID)] * 1000
    start = time.perf_counter()
    predict(batch)
    elapsed = time.perf_counter() - start
    check('1000 objects in under 1 second', elapsed < 1, f'{elapsed:.3f}s')


def check_serving_imports():
    print('\n-- Serving imports (fresh process)')
    code = ("import sys, src.predict; "
            "print(any(m in sys.modules for m in ('matplotlib', 'src.data', 'src.model', 'src.evaluate')))")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    check('src.predict loads no matplotlib or training modules', result.stdout.strip() == 'False',
          result.stdout.strip() or result.stderr.strip()[-200:])


def check_version_guard():
    print('\n-- scikit-learn version guard')
    real_version = sklearn.__version__
    load_model.cache_clear()
    sklearn.__version__ = '0.0.0'  # pretend a different version is installed
    try:
        load_model()
        check('refuses a model saved with another scikit-learn version', False, 'model loaded anyway')
    except RuntimeError as e:
        check('refuses a model saved with another scikit-learn version', True, str(e)[:60] + '...')
    finally:
        sklearn.__version__ = real_version
        load_model.cache_clear()


def main():
    check_examples()
    check_order_preserved()
    check_validation()
    check_edge_cases()
    check_serving_imports()
    check_version_guard()

    print(f'\n{len(failures)} failed' if failures else '\nAll checks passed')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
