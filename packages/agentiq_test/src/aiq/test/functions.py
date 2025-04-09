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

from collections.abc import AsyncGenerator

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.api_server import AIQChatRequest
from aiq.data_models.api_server import AIQChatResponse
from aiq.data_models.api_server import AIQChatResponseChunk
from aiq.data_models.function import FunctionBaseConfig


class EchoFunctionConfig(FunctionBaseConfig, name="test_echo"):
    use_openai_api: bool = False


@register_function(config_type=EchoFunctionConfig)
async def echo_function(config: EchoFunctionConfig, builder: Builder):

    async def inner(message: str) -> str:
        return message

    async def inner_oai(message: AIQChatRequest) -> AIQChatResponse:
        return AIQChatResponse.from_string(message.messages[0].content)

    if (config.use_openai_api):
        yield inner_oai
    else:
        yield inner


class StreamingEchoFunctionConfig(FunctionBaseConfig, name="test_streaming_echo"):
    use_openai_api: bool = False


@register_function(config_type=StreamingEchoFunctionConfig)
async def streaming_function(config: StreamingEchoFunctionConfig, builder: Builder):

    def oai_to_list(message: AIQChatRequest) -> list[str]:
        return [m.content for m in message.messages]

    async def inner(message: list[str]) -> AsyncGenerator[str]:
        for value in message:
            yield value

    async def inner_oai(message: AIQChatRequest) -> AsyncGenerator[AIQChatResponseChunk]:
        for value in oai_to_list(message):
            yield AIQChatResponseChunk.from_string(value)

    yield FunctionInfo.from_fn(inner_oai if config.use_openai_api else inner, converters=[oai_to_list])


class ConstantFunctionConfig(FunctionBaseConfig, name="test_constant"):
    response: str


@register_function(config_type=ConstantFunctionConfig)
async def constant_function(config: ConstantFunctionConfig, builder: Builder):

    async def inner() -> str:
        return config.response

    yield inner


class StreamingConstantFunctionConfig(FunctionBaseConfig, name="test_streaming_constant"):
    responses: list[str]


@register_function(config_type=StreamingConstantFunctionConfig)
async def streaming_constant_function(config: StreamingConstantFunctionConfig, builder: Builder):

    async def inner() -> AsyncGenerator[str]:
        for value in config.responses:
            yield value

    yield inner
