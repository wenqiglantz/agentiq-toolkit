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
import os

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import EmbedderRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class CarMaintenanceFunctionConfig(FunctionBaseConfig, name="car_maintenance"):
    llm_name: LLMRef
    embedding_name: EmbedderRef
    data_dir: str
    api_key: str | None = None

@register_function(config_type=CarMaintenanceFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LLAMA_INDEX])
async def car_maintenance_tool(tool_config: CarMaintenanceFunctionConfig, builder: Builder):
    """
    Create a car maintenance function that provides comprehensive car maintenance services including problem 
    diagnosis, cost estimation, maintenance scheduling, and calendar invites.

    Parameters
    ----------
    tool_config : CarMaintenanceFunctionConfig
        Configuration for the car maintenance function
    builder : Builder
        The AgentIQ builder instance

    Returns
    -------
    A FunctionInfo object that can generate car maintenance plans.
    """

    from colorama import Fore
    from llama_index.core import Settings
    from llama_index.core.agent import FunctionCallingAgentWorker
    from llama_index.vector_stores.faiss import FaissVectorStore
    from datetime import datetime, timedelta
    from llama_index.core.tools import FunctionTool
    from llama_index.core.agent import FunctionCallingAgentWorker
    import faiss, json
    
    if (not tool_config.api_key):
        tool_config.api_key = os.getenv("NVIDIA_API_KEY")

    if not tool_config.api_key:
        raise ValueError(
            "API token must be provided in the configuration or in the environment variable `NVIDIA_API_KEY`")

    logger.info("##### processing data from ingesting files in this folder : %s", tool_config.data_dir)

    # Get LLM from builder
    llm = await builder.get_llm(tool_config.llm_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)

    # Get embedder from builder
    embedder = await builder.get_embedder(tool_config.embedding_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)

    Settings.embed_model = embedder
    Settings.llm = llm

    # Vector store setup
    faiss_index_problems = faiss.IndexFlatL2(1024)
    problems_vector_store = FaissVectorStore(
        faiss_index=faiss_index_problems
    )

    faiss_index_parts = faiss.IndexFlatL2(1024)
    parts_vector_store = FaissVectorStore(
        faiss_index=faiss_index_parts
    )

    faiss_index_diagnostics = faiss.IndexFlatL2(1024)
    diagnostics_vector_store = FaissVectorStore(
        faiss_index=faiss_index_diagnostics
    )

    faiss_index_cost_estimates = faiss.IndexFlatL2(1024)
    cost_estimates_vector_store = FaissVectorStore(
        faiss_index=faiss_index_cost_estimates
    )

    faiss_index_maintenance_schedules = faiss.IndexFlatL2(1024)
    maintenance_schedules_vector_store = FaissVectorStore(
        faiss_index=faiss_index_maintenance_schedules
    )

    faiss_index_cars = faiss.IndexFlatL2(1024)
    cars_vector_store = FaissVectorStore(
        faiss_index=faiss_index_cars
    )

    # Load and index documents directly from file paths
    problems_index = load_and_index_document_from_file(tool_config.data_dir+"problems.json", problems_vector_store)
    parts_index = load_and_index_document_from_file(tool_config.data_dir+"parts.json", parts_vector_store)
    cars_index = load_and_index_document_from_file(tool_config.data_dir+"cars_models.json", cars_vector_store)
    diagnostics_index = load_and_index_document_from_file(tool_config.data_dir+"diagnostics.json", diagnostics_vector_store)
    cost_estimates_index = load_and_index_document_from_file(tool_config.data_dir+"cost_estimates.json", cost_estimates_vector_store)
    maintenance_schedules_index = load_and_index_document_from_file(tool_config.data_dir+"maintenance.json", maintenance_schedules_vector_store)

    # Create retrievers
    problems_retriever = create_retriever(problems_index)
    parts_retriever = create_retriever(parts_index)
    cars_retriever = create_retriever(cars_index)
    diagnostics_retriever = create_retriever(diagnostics_index)
    cost_estimates_retriever = create_retriever(cost_estimates_index)
    maintenance_schedules_retriever = create_retriever(maintenance_schedules_index)

    # Create function tools and set up agent
    def retrieve_problems(query: str) -> str:
        """Searches the problem catalog to find relevant automotive problems for the query."""
        docs = problems_retriever.retrieve(query)
        information = str([doc.text[:200]for doc in docs])
        return information

    def retrieve_parts(query: str) -> str:
        """Searches the parts catalog to find relevant parts for the query."""
        docs = parts_retriever.retrieve(query)
        information = str([doc.text[:200]for doc in docs])
        return information

    def diagnose_car_problem(symptoms: str) -> str:
        """Uses the diagnostics data to find potential causes for given symptoms."""
        docs = diagnostics_retriever.retrieve(symptoms)
        information = str([doc.text[:200]for doc in docs])
        return information

    def estimate_repair_cost(problem: str) -> str:
        """Provides a cost estimate for a given car problem or repair."""
        docs = cost_estimates_retriever.retrieve(problem)
        information = str([doc.text[:200]for doc in docs])
        return information

    def get_maintenance_schedule(mileage: int) -> str:
        """Retrieves the recommended maintenance schedule based on mileage."""
        docs = maintenance_schedules_retriever.retrieve(str(mileage))
        information = str([doc.text[:200]for doc in docs])
        return information

    def comprehensive_diagnosis(symptoms: str) -> str:
        """
        Provides a comprehensive diagnosis including possible causes, estimated costs, and required parts.

        Args:
            symptoms: A string describing the car's symptoms.

        Returns:
            A string with a comprehensive diagnosis report.
        """
        # Use existing tools
        possible_causes = diagnose_car_problem(symptoms)

        # Extract the most likely cause (this is a simplification)
        likely_cause = possible_causes[0] if possible_causes else "Unknown issue"

        estimated_cost = estimate_repair_cost(likely_cause)
        required_parts = retrieve_parts(likely_cause)

        report = f"Comprehensive Diagnosis Report:\n\n"
        report += f"Symptoms: {symptoms}\n\n"
        report += f"Possible Causes:\n{possible_causes}\n\n"
        report += f"Most Likely Cause: {likely_cause}\n\n"
        report += f"Estimated Cost:\n{estimated_cost}\n\n"
        report += f"Required Parts:\n{required_parts}\n\n"
        report += "Please note that this is an initial diagnosis. For accurate results, please consult with our professional mechanic."
        return report

    def get_car_model_info(car_make: str, car_model: str, car_year: int) -> dict:
        """Retrieve car model information from cars_models.json."""
        with open('data/cars_models.json', 'r') as file:
            car_models = json.load(file)

        for car in car_models:
            if (car['car_make'].lower() == car_make.lower() and car['car_model'].lower() == car_model.lower() and car['car_year'] == car_year):
                return car
        return {}

    def retrieve_car_details(make: str, model: str, year: int) -> str:
        """Retrieves the make, model, and year of the car and return the common issues if any."""
        car_details = get_car_model_info(make, model, year)  
        if car_details:
            return f"{year} {make} {model} - Common Issues: {', '.join(car_details['common_issues'])}"
        return f"{year} {make} {model} - No common issues found."

    def plan_maintenance(mileage: int, car_make: str, car_model: str, car_year: int) -> str:
        """
        Creates a comprehensive maintenance plan based on the car's mileage and details.

        Args:
            mileage: The current mileage of the car.
            car_make: The make of the car.
            car_model: The model of the car.
            car_year: The year the car was manufactured.

        Returns:
            A string with a comprehensive maintenance plan.
        """
        car_details = retrieve_car_details(car_make, car_model, car_year)
        car_model_info = get_car_model_info(car_make, car_model, car_year)

        plan = f"Maintenance Plan for {car_year} {car_make} {car_model} at {mileage} miles:\n\n"
        plan += f"Car Details: {car_details}\n\n"

        if car_model_info:
            plan += f"Common Issues:\n"
            for issue in car_model_info['common_issues']:
                plan += f"- {issue}\n"

            plan += f"\nEstimated Time: {car_model_info['estimated_time']}\n\n"
        else:
            plan += "No specific maintenance tasks found for this car model and mileage.\n\n"

        plan += "Please consult with our certified mechanic for a more personalized maintenance plan."

        return plan


    def create_calendar_invite(event_type: str, car_details: str, duration: int = 60) -> str:
        """
        Simulates creating a calendar invite for a car maintenance or repair event.

        Args:
            event_type: The type of event (e.g., "Oil Change", "Brake Inspection").
            car_details: Details of the car (make, model, year).
            duration: Duration of the event in minutes (default is 60).

        Returns:
            A string describing the calendar invite.
        """
        # Simulate scheduling the event for next week
        event_date = datetime.now() + timedelta(days=7)
        event_time = event_date.replace(hour=10, minute=0, second=0, microsecond=0)

        invite = f"Calendar Invite Created:\n\n"
        invite += f"Event: {event_type} for {car_details}\n"
        invite += f"Date: {event_time.strftime('%Y-%m-%d')}\n"
        invite += f"Time: {event_time.strftime('%I:%M %p')}\n"
        invite += f"Duration: {duration} minutes\n"
        invite += f"Location: Your Trusted Auto Shop, 123 Main St, San Francisco, California\n\n"

        return invite

    def coordinate_car_care(query: str, car_make: str, car_model: str, car_year: int, mileage: int) -> str:
        """
        Coordinates overall car care by integrating diagnosis, maintenance planning, and scheduling.

        Args:
            query: The user's query or description of the issue.
            car_make: The make of the car.
            car_model: The model of the car.
            car_year: The year the car was manufactured.
            mileage: The current mileage of the car.

        Returns:
            A string with a comprehensive car care plan.
        """
        car_details = retrieve_car_details(car_make, car_model, car_year)

        # Check if it's a problem or routine maintenance
        if "problem" in query.lower() or "issue" in query.lower():
            diagnosis = comprehensive_diagnosis(query)
            plan = f"Based on your query, here's a diagnosis:\n\n{diagnosis}\n\n"

            # Extract the most likely cause (this is a simplification)
            likely_cause = diagnosis.split("Most Likely Cause:")[1].split("\n")[0].strip()

            # Create a calendar invite for repair
            invite = create_calendar_invite(f"Repair: {likely_cause}", car_details)
            plan += f"I've prepared a calendar invite for the repair:\n\n{invite}\n\n"
        else:
            maintenance_plan = plan_maintenance(mileage, car_make, car_model, car_year)
            plan = f"Here's your maintenance plan:\n\n{maintenance_plan}\n\n"

            # Create a calendar invite for the next maintenance task
            next_task = maintenance_plan.split("Task:")[1].split("\n")[0].strip()
            invite = create_calendar_invite(f"Maintenance: {next_task}", car_details)
            plan += f"I've prepared a calendar invite for your next maintenance task:\n\n{invite}\n\n"

        plan += "Remember to consult with a professional mechanic for personalized advice and service."

        return plan


    ## Create function tools
    retrieve_problems_tool = FunctionTool.from_defaults(fn=retrieve_problems)
    retrieve_parts_tool = FunctionTool.from_defaults(fn=retrieve_parts)
    diagnostic_tool = FunctionTool.from_defaults(fn=diagnose_car_problem)
    cost_estimator_tool = FunctionTool.from_defaults(fn=estimate_repair_cost)
    maintenance_schedule_tool = FunctionTool.from_defaults(fn=get_maintenance_schedule)
    comprehensive_diagnostic_tool = FunctionTool.from_defaults(fn=comprehensive_diagnosis)
    maintenance_planner_tool = FunctionTool.from_defaults(fn=plan_maintenance)
    calendar_invite_tool = FunctionTool.from_defaults(fn=create_calendar_invite)
    car_care_coordinator_tool = FunctionTool.from_defaults(fn=coordinate_car_care)
    retrieve_car_details_tool = FunctionTool.from_defaults(fn=retrieve_car_details)

    tools = [
        retrieve_problems_tool,
        retrieve_parts_tool,
        diagnostic_tool,
        cost_estimator_tool,
        maintenance_schedule_tool,
        comprehensive_diagnostic_tool,
        maintenance_planner_tool,
        calendar_invite_tool,
        car_care_coordinator_tool,
        retrieve_car_details_tool
    ]

    # Create agent
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools, # type: ignore
        llm=llm,
        verbose=True,
    )
    agent = agent_worker.as_agent()

    async def _arun(inputs: str) -> str:
        agent_response = (await agent.achat(inputs))
        logger.info("response from car_maintenance : \n %s %s", Fore.MAGENTA, agent_response.response)
        output = agent_response.response
        return output

    # Create a Generic AgentIQ tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(_arun, description="extract relevant car maintenance data per user input query")


import json
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.faiss import FaissVectorStore 

def load_and_index_document_from_file(file_path: str, vector_store: FaissVectorStore) -> VectorStoreIndex:
    """Load a document from a single file and index it."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        document = Document(text=json.dumps(data))

    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = parser.get_nodes_from_documents([document])
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes, storage_context=storage_context)


def create_retriever(index: VectorStoreIndex) -> VectorIndexRetriever:
    """Create a retriever from the index."""
    return index.as_retriever(similarity_top_k=5) 
