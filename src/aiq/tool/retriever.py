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

import logging

from pydantic import BaseModel
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import RetrieverRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.retriever.interface import AIQRetriever
from aiq.retriever.models import RetrieverError
from aiq.retriever.models import RetrieverOutput

logger = logging.getLogger(__name__)


class AIQRetrieverConfig(FunctionBaseConfig, name="aiq_retriever"):
    """
    AIQRetriever tool which provides a common interface for different vectorstores. Its
    configuration uses clients, which are the vectorstore-specific implementaiton of the retriever interface.
    """
    retriever: RetrieverRef = Field(description="The retriever instance name from the workflow configuration object.")
    raise_errors: bool = Field(
        default=True,
        description="If true the tool will raise exceptions, otherwise it will log them as warnings and return []",
    )
    topic: str | None = Field(default=None, description="Used to provide a more detailed tool description to the agent")
    description: str | None = Field(default=None, description="If present it will be used as the tool description")


def _get_description_from_config(config: AIQRetrieverConfig) -> str:
    """
    Generate a description of what the tool will do based on how it is configured.
    """
    description = "Retrieve document chunks{topic} which can be used to answer the provided question."

    _topic = f" related to {config.topic}" if config.topic else ""

    return description.format(topic=_topic) if not config.description else config.description


@register_function(config_type=AIQRetrieverConfig)
async def aiq_retriever_tool(config: AIQRetrieverConfig, builder: Builder):
    """
    Configure an AgentIQ Retriever Tool which supports different clients such as Milvus and Nemo Retriever.

    Args:
        config: A config object with required parameters 'client' and 'client_config'
        builder: A workflow builder object
    """

    class RetrieverInputSchema(BaseModel):
        query: str = Field(description="The query to be searched in the configured data store")

    client: AIQRetriever = await builder.get_retriever(config.retriever)

    async def _retrieve(query: str) -> RetrieverOutput:
        try:
            retrieved_context = await client.search(query=query)
            logger.info("Retrieved %s records for query %s.", len(retrieved_context), query)
            return retrieved_context

        except RetrieverError as e:
            if config.raise_errors:
                raise e
            logger.warning("Retriever threw an error: %s. Returning an empty response.", e)
            return RetrieverOutput(results=[])

    yield FunctionInfo.from_fn(
        fn=_retrieve,
        input_schema=RetrieverInputSchema,
        description=_get_description_from_config(config),
    )
