"""API input validation: what POST /predict accepts and rejects, checked on the schemas directly."""
import pytest
from pydantic import ValidationError

from src.config import MAX_BATCH_SIZE, PLACEHOLDER
from src.schemas import PredictRequest, StellarObject

# A real STAR from the test set
VALID = {'u': 21.481, 'g': 20.110, 'r': 19.479, 'i': 19.276, 'z': 19.240}


def test_accepts_a_valid_object():
    assert StellarObject(**VALID).model_dump() == VALID


def test_accepts_a_number_sent_as_text():
    # JSON clients sometimes send "21.5"; pydantic turns it into the float 21.5
    assert StellarObject(**{**VALID, 'u': '21.5'}).u == 21.5


@pytest.mark.parametrize('fields', [
    pytest.param({**VALID, 'u': PLACEHOLDER}, id='placeholder -9999'),
    pytest.param({**VALID, 'u': float('nan')}, id='NaN'),
    pytest.param({**VALID, 'u': float('inf')}, id='infinity'),
    pytest.param({**VALID, 'u': float('-inf')}, id='minus infinity'),
    pytest.param({**VALID, 'u': 'abc'}, id='text'),
    pytest.param({**VALID, 'R': 19.5}, id='unknown field'),
    pytest.param({k: v for k, v in VALID.items() if k != 'z'}, id='missing field'),
])
def test_rejects_invalid_object(fields):
    with pytest.raises(ValidationError):
        StellarObject(**fields)


def test_placeholder_error_explains_why():
    with pytest.raises(ValidationError, match='placeholder for a missing measurement'):
        StellarObject(**{**VALID, 'g': PLACEHOLDER})


def test_request_accepts_one_object_and_a_full_batch():
    assert len(PredictRequest(objects=[VALID]).objects) == 1
    assert len(PredictRequest(objects=[VALID] * MAX_BATCH_SIZE).objects) == MAX_BATCH_SIZE


@pytest.mark.parametrize('body', [
    pytest.param({'objects': []}, id='empty list'),
    pytest.param({'objects': [VALID] * (MAX_BATCH_SIZE + 1)}, id='over the batch limit'),
    pytest.param({}, id='no objects key'),
    pytest.param({'objects': [VALID], 'extra': 1}, id='unknown top-level field'),
])
def test_request_rejects_invalid_body(body):
    with pytest.raises(ValidationError):
        PredictRequest(**body)
