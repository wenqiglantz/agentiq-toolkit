# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import typing
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from pydantic import Discriminator
from pydantic import FilePath
from pydantic import Tag

from aiq.data_models.common import BaseModelRegistryTag
from aiq.data_models.common import TypedBaseModel


class EvalS3Config(BaseModel):

    endpoint_url: str
    bucket: str
    access_key: str
    secret_key: str


class EvalFilterEntryConfig(BaseModel):
    # values are lists of allowed/blocked values
    field: dict[str, list[str | int | float]] = {}


class EvalFilterConfig(BaseModel):
    allowlist: EvalFilterEntryConfig | None = None
    denylist: EvalFilterEntryConfig | None = None


class EvalDatasetStructureConfig(BaseModel):
    disable: bool = False
    question_key: str = "question"
    answer_key: str = "answer"
    generated_answer_key: str = "generated_answer"
    trajectory_key: str = "intermediate_steps"
    expected_trajectory_key: str = "expected_intermediate_steps"


# Base model
class EvalDatasetBaseConfig(TypedBaseModel, BaseModelRegistryTag):

    id_key: str = "id"
    structure: EvalDatasetStructureConfig = EvalDatasetStructureConfig()

    # Filters
    filter: EvalFilterConfig | None = EvalFilterConfig()

    s3: EvalS3Config | None = None

    remote_file_path: str | None = None  # only for s3
    file_path: Path | str = Path(".tmp/aiq/examples/default/default.json")


class EvalDatasetJsonConfig(EvalDatasetBaseConfig, name="json"):

    @staticmethod
    def parser() -> tuple[Callable, dict]:
        return pd.read_json, {}


def read_jsonl(file_path: FilePath, **kwargs):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


class EvalDatasetJsonlConfig(EvalDatasetBaseConfig, name="jsonl"):

    @staticmethod
    def parser() -> tuple[Callable, dict]:
        return read_jsonl, {}


class EvalDatasetCsvConfig(EvalDatasetBaseConfig, name="csv"):

    @staticmethod
    def parser() -> tuple[Callable, dict]:
        return pd.read_csv, {}


class EvalDatasetParquetConfig(EvalDatasetBaseConfig, name="parquet"):

    @staticmethod
    def parser() -> tuple[Callable, dict]:
        return pd.read_parquet, {}


class EvalDatasetXlsConfig(EvalDatasetBaseConfig, name="xls"):

    @staticmethod
    def parser() -> tuple[Callable, dict]:
        return pd.read_excel, {"engine": "openpyxl"}


# Union model with discriminator
EvalDatasetConfig = typing.Annotated[typing.Annotated[EvalDatasetJsonConfig, Tag(EvalDatasetJsonConfig.static_type())]
                                     | typing.Annotated[EvalDatasetCsvConfig, Tag(EvalDatasetCsvConfig.static_type())]
                                     | typing.Annotated[EvalDatasetXlsConfig, Tag(EvalDatasetXlsConfig.static_type())]
                                     | typing.Annotated[EvalDatasetParquetConfig,
                                                        Tag(EvalDatasetParquetConfig.static_type())]
                                     | typing.Annotated[EvalDatasetJsonlConfig,
                                                        Tag(EvalDatasetJsonlConfig.static_type())],
                                     Discriminator(TypedBaseModel.discriminator)]
