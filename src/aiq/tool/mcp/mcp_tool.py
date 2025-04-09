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
from pydantic import HttpUrl

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class MCPToolConfig(FunctionBaseConfig, name="mcp_tool_wrapper"):
    """
    Function which connects to a Model Context Protocol (MCP) server and wraps the selected tool as an AgentIQ function.
    """
    # Add your custom configuration parameters here
    url: HttpUrl = Field(description="The URL of the MCP server")
    mcp_tool_name: str = Field(description="The name of the tool served by the MCP Server that you want to use")
    description: str | None = Field(default=None,
                                    description="""
        Description for the tool that will override the description provided by the MCP server. Should only be used if
        the description provided by the server is poor or nonexistent
        """)


@register_function(config_type=MCPToolConfig)
async def mcp_tool(config: MCPToolConfig, builder: Builder):
    """
    Generate an AgentIQ Function that wraps a tool provided by the MCP server.
    """

    from aiq.tool.mcp.mcp_client import MCPBuilder
    from aiq.tool.mcp.mcp_client import MCPToolClient

    client = MCPBuilder(url=str(config.url))

    tool: MCPToolClient = await client.get_tool(config.mcp_tool_name)
    if config.description:
        tool.set_description(description=config.description)

    logger.info("Configured to use tool: %s from MCP server at %s", tool.name, str(config.url))

    def _convert_from_str(input_str: str) -> tool.input_schema:
        return tool.input_schema.model_validate_json(input_str)

    async def _response_fn(tool_input: BaseModel | None = None, **kwargs) -> str:
        if tool_input:
            args = tool_input.model_dump()
            return await tool.acall(args)

        _ = tool.input_schema.model_validate(kwargs)
        return await tool.acall(kwargs)

    yield FunctionInfo.create(single_fn=_response_fn,
                              description=tool.description,
                              input_schema=tool.input_schema,
                              converters=[_convert_from_str])
