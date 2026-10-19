"""Explicit bilateral completion of a unilateral movement, not admission."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


class UnilateralPrescription(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True,
                              extra='forbid', frozen=True, strict=True)
    quantity_mode: Literal['repsPerSide', 'durationPerSide']
    side_count: Literal[2] = 2
    set_unit: Literal['bothSides'] = 'bothSides'
    side_order: Literal['leftThenRight', 'rightThenLeft'] = 'leftThenRight'
    side_switch_sec: int = Field(ge=0, le=60)
    seconds_per_rep: int | None = Field(default=None, ge=1, le=10)
    rest_placement: Literal['afterBothSides'] = 'afterBothSides'

    @model_validator(mode='after')
    def validate_units(self):
        if (self.quantity_mode == 'repsPerSide') != (self.seconds_per_rep is not None):
            raise ValueError('Repetition mode requires secondsPerRep; duration mode forbids it')
        return self
