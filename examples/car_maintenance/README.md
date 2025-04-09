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


# Car Maintenance

This example demonstrates how to create an AI agent that can diagnose car issues, suggest fixes, and manage maintenance schedules. It combines LlamaIndex, NVIDIA NIMs, and FAISS to build a sophisticated AI system that goes beyond basic information retrieval.  This notebook was inspired by the LanceDB blog post [Multi document agentic RAG: A walkthrough](https://blog.lancedb.com/multi-document-agentic-rag-a-walkthrough/).

## Table of Contents

- [Car Maintenance](#car-maintenance)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Installation and Setup](#installation-and-setup)
    - [Setup Virtual Environment and Install AgentIQ](#setup-virtual-environment-and-install-agentiq)
    - [Install this Workflow:](#install-this-workflow)
    - [Set Up API Keys](#set-up-api-keys)
  - [Example Usage](#example-usage)
    - [Run the Workflow](#run-the-workflow)
  - [Deployment-Oriented Setup](#deployment-oriented-setup)
    - [Build the Docker Image](#build-the-docker-image)
    - [Run the Docker Container](#run-the-docker-container)
    - [Test the API](#test-the-api)
    - [Expected API Output](#expected-api-output)


## Key Features

- **Pre-built Tools:** Leverages core AgentIQ library agent and tools.
- **ReAct Agent:** Performs reasoning between tool call; utilizes tool names and descriptions to appropriately route to the correct tool
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LlamaIndex tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.


## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/intro/install.md) to create the development environment and install AgentIQ.

### Install this Workflow:

From the root directory of the AgentIQ library, run the following commands:

```bash
uv pip install -e examples/car_maintenance
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../docs/source/intro/get-started.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Example Usage

### Run the Workflow

Run the following command from the root of the AgentIQ repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/car_maintenance/src/car_maintenance/configs/config.yml --input "My car has 60,000 miles on it. What maintenance should I be doing now, and how much will it cost?"
```

**Expected Output**
```console
(.venv) wglantz@NV-D3939Y3:~/agentiq$ aiq run --config_file examples/car_maintenance/src/car_maintenance/configs/config.yml --input "My car has 60,000 miles on it. What maintenance should I be doing now, and how much will it cost?"
2025-03-21 11:11:54,261 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (568.838358 ms). Ensure all imports are inside your registered functions.
2025-03-21 11:11:56,018 - aiq.runtime.loader - WARNING - Loading module 'car_maintenance.register' from entry point 'car_maintenance' took a long time (1720.917940 ms). Ensure all imports are inside your registered functions.
2025-03-21 11:11:56,052 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples/car_maintenance/src/car_maintenance/configs/config.yml'
2025-03-21 11:11:56,056 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-03-21 11:11:56,079 - aiq.profiler.decorators - INFO - LlamaIndex callback handler registered
2025-03-21 11:11:56,166 - faiss.loader - INFO - Loading faiss with AVX2 support.
2025-03-21 11:11:56,186 - faiss.loader - INFO - Successfully loaded faiss with AVX2 support.
2025-03-21 11:11:56,189 - car_maintenance.car_maintenance_function - INFO - ##### processing data from ingesting files in this folder : ./examples/car_maintenance/src/car_maintenance/data/
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
2025-03-21 11:11:58,836 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2025-03-21 11:11:58,997 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2025-03-21 11:11:59,139 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2025-03-21 11:11:59,220 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2025-03-21 11:11:59,326 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2025-03-21 11:11:59,409 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2025-03-21 11:11:59,428 - aiq.profiler.decorators - INFO - Langchain callback handler registered
2025-03-21 11:11:59,485 - aiq.agent.tool_calling_agent.agent - INFO - Initialized Tool Calling Agent Graph
2025-03-21 11:11:59,489 - aiq.agent.tool_calling_agent.agent - INFO - Tool Calling Agent Graph built and compiled successfully

Configuration Summary:
--------------------
Workflow Type: tool_calling_agent
Number of Functions: 1
Number of LLMs: 1
Number of Embedders: 1
Number of Memory: 0
Number of Retrievers: 0

2025-03-21 11:11:59,491 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('My car has 60,000 miles on it. What maintenance should I be doing now, and how much will it cost?',)
2025-03-21 11:11:59,495 - aiq.agent.tool_calling_agent.agent - INFO - Calling agent
2025-03-21 11:12:06,689 - aiq.agent.tool_calling_agent.agent - INFO - Calling tools: ['car_maintenance']
Added user message to memory: car maintenance at 60,000 miles and cost
2025-03-21 11:12:07,829 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
=== Calling Function ===
Calling function: get_maintenance_schedule with args: {"mileage": 60000}
2025-03-21 11:12:07,936 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
=== Function Output ===
['[{"mileage": 30000, "tasks": ["Oil and filter change", "Tire rotation", "Air filter replacement", "Brake inspection"], "importance": "Regular maintenance", "estimated_time": "2-3 hours"}, {"mileage": ', '{"mileage": 510000, "tasks": ["Exhaust system replacement", "Suspension system replacement", "Clutch replacement", "Differential fluid change"], "importance": "Major service", "estimated_time": "6-8 h']
2025-03-21 11:12:31,333 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
=== LLM Response ===
<|python_tag|><function>estimate_repair_cost">{"problem": "Oil and filter change, Tire rotation, Air filter replacement, Brake inspection"}</function>
2025-03-21 11:12:31,338 - car_maintenance.car_maintenance_function - INFO - response from car_maintenance : 
  <|python_tag|><function>estimate_repair_cost">{"problem": "Oil and filter change, Tire rotation, Air filter replacement, Brake inspection"}</function>
2025-03-21 11:12:31,340 - aiq.agent.tool_calling_agent.agent - INFO - Calling agent
2025-03-21 11:12:42,287 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-03-21 11:12:42,288 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
["Based on your car's mileage, it is recommended to perform the following maintenance tasks:\n\n1. Oil and filter change\n2. Tire rotation\n3. Air filter replacement\n4. Brake inspection\n\nThe estimated cost for these tasks is:\n\n* Oil and filter change: $75-$150\n* Tire rotation: $20-$50\n* Air filter replacement: $15-$30\n* Brake inspection: $20-$50\n\nTotal estimated cost: $130-$330\n\nPlease note that these estimates are approximate and may vary depending on the make and model of your car, as well as the labor rates of the mechanic or repair shop. It's always a good idea to consult with a professional mechanic for a more accurate estimate."]
--------------------------------------------------
```
---

## Deployment-Oriented Setup

For a production deployment, use Docker:

### Build the Docker Image

Prior to building the Docker image ensure that you have followed the steps in the [Installation and Setup](#installation-and-setup) section, and you are currently in the AgentIQ virtual environment.

From the root directory of the Simple Calculator repository, build the Docker image:

```bash
docker build --build-arg AIQ_VERSION=$(python -m setuptools_scm) -t car_maintenance -f examples/car_maintenance/Dockerfile .
```

### Run the Docker Container
Deploy the container:

```bash
docker run --rm -p 8000:8000 -e NVIDIA_API_KEY car_maintenance
```

### Test the API
Use the following curl command to test the deployed API:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"input_message": "My car has 60,000 miles on it. What maintenance should I be doing now, and how much will it cost?"}'
  ```

### Expected API Output
The API response should look like this:

```bash
{"value":"Based on your car's mileage, it is recommended to perform the following maintenance tasks:\n\n1. Oil and filter change\n2. Tire rotation\n3. Air filter replacement\n4. Brake inspection\n\nThe estimated cost for these tasks is:\n\n* Oil and filter change: $75-$150\n* Tire rotation: $20-$50\n* Air filter replacement: $15-$30\n* Brake inspection: $20-$50\n\nTotal estimated cost: $130-$330\n\nPlease note that these estimates are approximate and may vary depending on the make and model of your car, as well as the labor rates of the mechanic or repair shop. It's always a good idea to consult with a professional mechanic for a more accurate estimate."}
```
