# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import EmbedderRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class WebQueryToolConfig(FunctionBaseConfig, name="webpage_query"):
    webpage_url: str
    description: str
    chunk_size: int = 1024
    embedder_name: EmbedderRef = "nvidia/nv-embedqa-e5-v5"


@register_function(config_type=WebQueryToolConfig)
async def webquery_tool(config: WebQueryToolConfig, builder: Builder):

    from langchain.tools.retriever import create_retriever_tool
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.embeddings import Embeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    logger.info("Generating docs for the webpage: %s", config.webpage_url)

    embeddings: Embeddings = await builder.get_embedder(config.embedder_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    loader = WebBaseLoader(config.webpage_url)

    # Cant use `aload` because its implemented incorrectly and is not async
    docs = [document async for document in loader.alazy_load()]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size)
    documents = text_splitter.split_documents(docs)
    vector = await FAISS.afrom_documents(documents, embeddings)

    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "webpage_search",
        config.description,
    )

    async def _inner(query: str) -> str:

        return await retriever_tool.arun(query)

    yield FunctionInfo.from_fn(_inner, description=config.description)
