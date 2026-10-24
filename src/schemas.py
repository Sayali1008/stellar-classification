from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import MAX_BATCH_SIZE, PLACEHOLDER


class StellarObject(BaseModel):
    u: float
    g: float
    r: float
    i: float
    z: float

    # reject unknown fields and reject NaN and infinite values on float fields
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    @field_validator('u', 'g', 'r', 'i', 'z')
    @classmethod
    def placeholder_not_allowed(cls, value: float) -> float:
        """
        Validate all the float fields and reject placeholders if they come in.
        """
        if value == PLACEHOLDER:
            raise ValueError(f'{PLACEHOLDER} is a placeholder for a missing measurement. This is not a real value.')
        return value


class Prediction(BaseModel):
    predicted_class: str
    probabilities: dict[str, float] # One probability per class, by class name; they should sum to 1
    # One message per filter outside the training range: the answer is less trustworthy there
    warnings: list[str] = []


class PredictRequest(BaseModel):
    """Body of POST /predict: always a list, so one object and a batch use the same shape."""
    model_config = ConfigDict(extra='forbid')

    objects: list[StellarObject] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class PredictResponse(BaseModel):
    """Answer from POST /predict: one prediction per object, in the same order, plus which model made them."""
    model_version: str
    predictions: list[Prediction]
