from typing import List

from pydantic import BaseModel, validator

from .common import Language

SUPPORTED_GENDERS = {'male', 'female'}


class Sentence(BaseModel):
    source: str

    # @validator('source', pre=True)
    # def blank_string_in_source(cls, value, field):
    #     if value == "":
    #         raise ValueError('source cannot be empty')
    #     return value


class TTSConfig(BaseModel):
    language: Language
    gender: str

    # @validator('gender', pre=True)
    # def blank_string_in_gender(cls, value, field):
    #     if value == "":
    #         raise ValueError('gender cannot be empty')
    #     if value not in SUPPORTED_GENDERS:
    #         raise ValueError('Unsupported gender value')
    #     return value


class TTSRequest(BaseModel):
    input: List[Sentence]
    config: TTSConfig

    # @validator('input', pre=True)
    # def input_cannot_be_empty(cls, value, field):
    #     if len(value) < 1:
    #         raise ValueError('input cannot be empty')
    #     return value
