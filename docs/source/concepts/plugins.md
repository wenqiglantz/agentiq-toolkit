<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Plugin System in NVIDIA AgentIQ

AgentIQ has a very extensible plugin system that allows you to add new tools, agents, workflows and more to the library. The plugin system is designed to be easy to use and allow developers to extend the library to their needs.

The plugin system is designed around two main concepts:

- **Entry Points**: Python entry points allow AgentIQ to discover plugins from any installed distribution package in a Python environment.
- **Decorators**: Decorators allow developers register their plugins with library.

These two concepts allow the library to be extended by installing any compatible plugins from a Python package index. Once installed, the plugin will be automatically discovered and loaded by AgentIQ.

AgentIQ utilizes the this plugin system for all first party components. This allows the library to be modular and extendable by default. Plugins from external libraries are treated exactly the same as first party plugins.


## Supported Plugins

AgentIQ currently supports the following plugins:

- **Embedder Clients**: Embedder Clients are implementations of embedder providers which are specific to a LLM framework. For example, when using the OpenAI embedder provider with the LangChain framework, the a LangChain OpenAI embedder client needs to be registered. To register an embedder client, you can use the {py:deco}`~aiq.cli.register_workflow.register_embedder_client` decorator.
- **Embedder Providers**: Embedder Providers are services which provide a way to embed text. For example, OpenAI and NVIDIA NIMs are embedder providers. To register an embedder provider, you can use the {py:deco}`~aiq.cli.register_workflow.register_embedder_provider` decorator.
- **Evaluators**: Evaluators are used by the evaluation framework to evaluate the performance of AgentIQ workflows. To register an evaluator, you can use the {py:deco}`~aiq.cli.register_workflow.register_evaluator` decorator.
- **Front Ends**: Front ends are the mechanism by which AgentIQ workflows are executed. Examples of front ends include a FastAPI server or a CLI. To register a front end, you can use the {py:deco}`~aiq.cli.register_workflow.register_front_end` decorator.
- **Functions**: Functions are one of the core building blocks of AgentIQ. They are used to define the tools and agents that can be used in a workflow. To register a function, you can use the {py:deco}`~aiq.cli.register_workflow.register_function` decorator.
- **LLM Clients**: LLM Clients are implementations of LLM providers which are specific to a LLM framework. For example, when using the NVIDIA NIMs LLM provider with the LangChain framework, the a NVIDIA LangChain LLM client needs to be registered. To register an LLM client, you can use the {py:deco}`~aiq.cli.register_llm_client` decorator.
- **LLM Providers**: An LLM provider is a service which provides a way to interact with an LLM. For example, OpenAI and NVIDIA NIMs are LLM providers. To register an LLM provider, you can use the {py:deco}`~aiq.cli.register_workflow.register_llm_provider` decorator.
- **Logging Methods**: Logging methods control the destination and format of log messages. To register a logging method, you can use the {py:deco}`~aiq.cli.register_workflow.register_logging_method` decorator.
- **Memory**: Memory plugins are used to store and retrieve information from a database to be used by an LLM. Examples of memory plugins include Zep and Mem0. To register a memory plugin, you can use the {py:deco}`~aiq.cli.register_workflow.register_memory` decorator.
- **Registry Handlers**: Registry handlers are used to register custom agent registries with AgentIQ. An agent registry is a collection of tools, agents, and workflows that can be used in a workflow. To register a registry handler, you can use the {py:deco}`~aiq.cli.register_workflow.register_registry_handler` decorator.
- **Retriever Clients**: Retriever clients are implementations of retriever providers which are specific to a LLM framework. For example, when using the Milvus retriever provider with the LangChain framework, the a LangChain Milvus retriever client needs to be registered. To register a retriever client, you can use the {py:deco}`~aiq.cli.register_workflow.register_retriever_client` decorator.
- **Retriever Providers**: Retriever providers are services which provide a way to retrieve information from a database. Examples of retriever providers include Chroma and Milvus. To register a retriever provider, you can use the {py:deco}`~aiq.cli.register_workflow.register_retriever_provider` decorator.
- **Telemetry Exporters**: Telemetry exporters send telemetry data to a telemetry service. To register a telemetry exporter, you can use the {py:deco}`~aiq.cli.register_workflow.register_telemetry_exporter` decorator.
- **Tool Wrappers**: Tool wrappers are used to wrap functions in a way that is specific to a LLM framework. For example, when using the LangChain framework, AgentIQ functions need to be wrapped in `BaseTool` class to be compatible with LangChain. To register a tool wrapper, you can use the {py:deco}`~aiq.cli.register_workflow.register_tool_wrapper` decorator.


## Framework Integrations

To keep the library lightweight, many of the first party plugins supported by AgentIQ are located in separate distribution packages. For example, the `agentiq-langchain` distribution contains all the LangChain specific plugins and the `agentiq-mem0ai` distribution contains the Mem0 specific plugins.

To install these first-party plugin libraries, you can use the full distribution name (for example, `agentiq-langchain`) or use the `agentiq[langchain]` extra distribution. A full list of the supported extras is listed below:

- `agentiq[crewai]` or `agentiq-crewai` - CrewAI specific plugins
- `agentiq[langchain]` or `agentiq-langchain` - LangChain specific plugins
- `agentiq[llama-index]` or `agentiq-llama-index` - LlamaIndex specific plugins
- `agentiq[mem0ai]` or `agentiq-mem0ai` - Mem0AI specific plugins
- `agentiq[semantic-kernel]` or `agentiq-semantic-kernel` - Semantic Kernel specific plugins
- `agentiq[test]` or `agentiq-test` - Test specific plugins
- `agentiq[zep-cloud]` or `agentiq-zep-cloud` - Zep Cloud specific plugins


## Anatomy of a Plugin

### Decorators

Registering a plugin with the library is done using decorators. Each plugin type has its own decorator that is used to register the plugin with the library. Once the decorator is loaded by python, it will be ready to use in the library.

The general format for a plugin decorator is:

```python
@register_<plugin_type>()
async def my_plugin_function(plugin_config: <plugin_config_type>, builder: Builder):

   # Execute any setup code needed

   # Yield the plugin which will be used by the library
   yield <plugin_type>

   # Execute any teardown code needed
```

All plugin decorators are async context managers. This allows the plugin to execute any setup and teardown code needed.

An example of a plugin decorator for the LangChain LLM client for OpenAI is:

```python
@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def openai_langchain(llm_config: OpenAIModelConfig, builder: Builder):

    from langchain_openai import ChatOpenAI

    yield ChatOpenAI(**llm_config.model_dump(exclude={"type"}, by_alias=True))
```

### Entry Point

Determining which plugins are available in a given environment is done through the use of [python entry points](https://packaging.python.org/en/latest/specifications/entry-points/). In AgentIQ, we scan the python environment for entry points which have the name `agentiq.components`. The value of the entry point is a python module that will be imported when the entry point is loaded.

For example, the `agentiq-langchain` distribution has the following entry point specified in the `pyproject.toml` file:

```toml
[project.entry-points.'aiq.components']
aiq_langchain = "aiq.plugins.langchain.register"
```

What this means is that when the `agentiq-langchain` distribution is installed, the `aiq.plugins.langchain.register` module will be imported when the entry point is loaded. This module must contain all the `@register_<plugin_type>` decorators which need to be loaded when the library is initialized.

> [!NOTE]
> The above syntax in the `pyproject.toml` file is specific to [uv](https://docs.astral.sh/uv/concepts/projects/config/#plugin-entry-points). Other package managers may have a different syntax for specifying entry points.


#### Multiple Plugins in a Single Distribution

It is possible to have multiple plugins in a single distribution. For example, the `agentiq-langchain` distribution contains both the LangChain LLM client and the LangChain embedder client.

To register multiple plugins in a single distribution, there are two options:

1. Register all plugins in a single module which imports all the plugins.
   1. This is the preferred method as it is more readable and easier to maintain.
   2. For example, if you have a `register.py` module in a package called `my_plugin`, your `register.py` module can do the following:

      ```python
      from .llm import register_llm_client
      from .embedder import register_embedder_client
      ```

2. Use multiple entry points to register all the plugins.
   1. This method is functionally equivalent to the first method, but requires re-installing the distribution to reflect changes to the plugins.
   2. For example, you could have two entry points in the `pyproject.toml` file:`

      ```toml
      [project.entry-points.'aiq.components']
      aiq_langchain = "aiq.plugins.langchain.register"
      aiq_langchain_tools = "aiq.plugins.langchain.tools.register"
      ```
