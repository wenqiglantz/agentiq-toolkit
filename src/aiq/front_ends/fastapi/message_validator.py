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

import datetime
import json
import logging
import uuid
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ValidationError

from aiq.data_models.api_server import AIQChatResponse
from aiq.data_models.api_server import AIQChatResponseChunk
from aiq.data_models.api_server import AIQResponseIntermediateStep
from aiq.data_models.api_server import AIQResponsePayloadOutput
from aiq.data_models.api_server import Error
from aiq.data_models.api_server import ErrorTypes
from aiq.data_models.api_server import SystemIntermediateStepContent
from aiq.data_models.api_server import SystemResponseContent
from aiq.data_models.api_server import TextContent
from aiq.data_models.api_server import WebSocketMessageStatus
from aiq.data_models.api_server import WebSocketMessageType
from aiq.data_models.api_server import WebSocketSystemInteractionMessage
from aiq.data_models.api_server import WebSocketSystemIntermediateStepMessage
from aiq.data_models.api_server import WebSocketSystemResponseTokenMessage
from aiq.data_models.api_server import WebSocketUserInteractionResponseMessage
from aiq.data_models.api_server import WebSocketUserMessage
from aiq.data_models.api_server import WorkflowSchemaType
from aiq.data_models.interactive import BinaryHumanPromptOption
from aiq.data_models.interactive import HumanPrompt
from aiq.data_models.interactive import HumanPromptBase
from aiq.data_models.interactive import HumanPromptBinary
from aiq.data_models.interactive import HumanPromptCheckbox
from aiq.data_models.interactive import HumanPromptDropdown
from aiq.data_models.interactive import HumanPromptRadio
from aiq.data_models.interactive import HumanPromptText
from aiq.data_models.interactive import HumanResponse
from aiq.data_models.interactive import HumanResponseBinary
from aiq.data_models.interactive import HumanResponseCheckbox
from aiq.data_models.interactive import HumanResponseDropdown
from aiq.data_models.interactive import HumanResponseRadio
from aiq.data_models.interactive import HumanResponseText
from aiq.data_models.interactive import MultipleChoiceOption

logger = logging.getLogger(__name__)


class MessageValidator:

    def __init__(self):
        self._message_type_schema_mapping: dict[str, type[BaseModel]] = {
            WebSocketMessageType.USER_MESSAGE: WebSocketUserMessage,
            WebSocketMessageType.RESPONSE_MESSAGE: WebSocketSystemResponseTokenMessage,
            WebSocketMessageType.INTERMEDIATE_STEP_MESSAGE: WebSocketSystemIntermediateStepMessage,
            WebSocketMessageType.SYSTEM_INTERACTION_MESSAGE: WebSocketSystemInteractionMessage,
            WebSocketMessageType.USER_INTERACTION_MESSAGE: WebSocketUserInteractionResponseMessage,
            WebSocketMessageType.ERROR_MESSAGE: Error
        }
        self._data_type_schema_mapping: dict[str, type[BaseModel]] = {
            WorkflowSchemaType.GENERATE: AIQResponsePayloadOutput,
            WorkflowSchemaType.CHAT: AIQChatResponse,
            WorkflowSchemaType.CHAT_STREAM: AIQChatResponseChunk,
            WorkflowSchemaType.GENERATE_STREAM: AIQResponseIntermediateStep,
        }
        self._message_parent_id: str = "default_id"

    async def validate_message(self, message: dict[str, Any]) -> BaseModel:
        """
        Validates an incoming WebSocket message against its expected schema.
        If validation fails, returns a system response error message.

        :param message: Incoming WebSocket message as a dictionary.
        :return: A validated Pydantic model.
        """
        validated_message: BaseModel

        try:
            message_type = message.get("type")
            if not message_type:
                raise ValueError(f"Missing message type: {json.dumps(message)}")

            schema: type[BaseModel] = await self.get_message_schema_by_type(message_type)

            if issubclass(schema, Error):
                raise TypeError(
                    f"An error was encountered processing an incoming WebSocket message of type: {message_type}")

            validated_message = schema(**message)
            return validated_message

        except (ValidationError, TypeError, ValueError) as e:
            logger.error("A data validation error %s occurred for message: %s", str(e), str(message), exc_info=True)
            return await self.create_system_response_token_message(message_type=WebSocketMessageType.ERROR_MESSAGE,
                                                                   content=Error(code=ErrorTypes.INVALID_MESSAGE,
                                                                                 message="Error validating message.",
                                                                                 details=str(e)))

    async def get_message_schema_by_type(self, message_type: str) -> type[BaseModel]:
        """
        Retrieves the corresponding Pydantic model schema based on the message type.

        :param message_type: The type of message as a string.
        :return: A Pydantic schema class if found, otherwise None.
        """
        try:
            schema: type[BaseModel] | None = self._message_type_schema_mapping.get(message_type)

            if schema is None:
                raise ValueError(f"Unknown message type: {message_type}")

            return schema

        except (TypeError, ValueError) as e:
            logger.error("Error retrieving schema for message type '%s': %s", message_type, str(e), exc_info=True)
            return Error

    async def convert_data_to_message_content(self, data_model: BaseModel) -> BaseModel:
        """
        Converts a Pydantic data model to a WebSocket message content instance.

        :param data_model: Pydantic Data Model instance.
        :return: A WebSocket Message Content Data Model instance.
        """

        validated_message_content: BaseModel = None
        try:
            if (isinstance(data_model, AIQResponsePayloadOutput)):
                validated_message_content = SystemResponseContent(text=data_model.payload)

            elif (isinstance(data_model, (AIQChatResponse, AIQChatResponseChunk))):
                validated_message_content = SystemResponseContent(text=data_model.choices[0].message.content)

            elif (isinstance(data_model, AIQResponseIntermediateStep)):
                validated_message_content = SystemIntermediateStepContent(name=data_model.name,
                                                                          payload=data_model.payload)
            elif (isinstance(data_model, HumanPromptBase)):
                validated_message_content = data_model
            elif (isinstance(data_model, SystemResponseContent)):
                return data_model
            else:
                raise ValueError(
                    f"Input data could not be converted to validated message content: {data_model.model_dump_json()}")

            return validated_message_content

        except ValueError as e:
            logger.error("Input data could not be converted to validated message content: %s", str(e), exc_info=True)
            return Error(code=ErrorTypes.INVALID_DATA_CONTENT, message="Input data not supported.", details=str(e))

    async def convert_text_content_to_human_response(self, text_content: TextContent,
                                                     human_prompt: HumanPromptBase) -> HumanResponse:
        """
        Converts Message Text Content data model to a Human Response Base data model instance.

        :param text_content: Pydantic TextContent Data Model instance.
        :param human_prompt: Pydantic HumanPrompt Data Model instance.
        :return: A Human Response Data Model instance.
        """

        human_response: HumanResponse = None
        try:
            if (isinstance(human_prompt, HumanPromptText)):
                human_response = HumanResponseText(text=text_content.text)

            elif (isinstance(human_prompt, HumanPromptBinary)):
                human_response = HumanResponseBinary(selected_option=BinaryHumanPromptOption(value=text_content.text))

            elif (isinstance(human_prompt, HumanPromptRadio)):
                human_response = HumanResponseRadio(selected_option=MultipleChoiceOption(value=text_content.text))

            elif (isinstance(human_prompt, HumanPromptCheckbox)):
                human_response = HumanResponseCheckbox(selected_option=MultipleChoiceOption(value=text_content.text))

            elif (isinstance(human_prompt, HumanPromptDropdown)):
                human_response = HumanResponseDropdown(selected_option=MultipleChoiceOption(value=text_content.text))
            else:
                raise ValueError("Message content type not found")

            return human_response

        except ValueError as e:
            logger.error("Error human response content not found: %s", str(e), exc_info=True)
            return HumanResponseText(text=str(e))

    async def resolve_message_type_by_data(self, data_model: BaseModel) -> str:
        """
        Resolve message type from a validated model

        :param data_model: Pydantic Data Model instance.
        :return: A WebSocket Message Content Data Model instance.
        """

        validated_message_type: str = ""
        try:
            if (isinstance(data_model, (AIQResponsePayloadOutput, AIQChatResponse, AIQChatResponseChunk))):
                validated_message_type = WebSocketMessageType.RESPONSE_MESSAGE

            elif (isinstance(data_model, AIQResponseIntermediateStep)):
                validated_message_type = WebSocketMessageType.INTERMEDIATE_STEP_MESSAGE

            elif (isinstance(data_model, HumanPromptBase)):
                validated_message_type = WebSocketMessageType.SYSTEM_INTERACTION_MESSAGE
            else:
                raise ValueError("Data type not found")

            return validated_message_type

        except ValueError as e:
            logger.error("Error type not found converting data to validated websocket message content: %s",
                         str(e),
                         exc_info=True)
            return WebSocketMessageType.ERROR_MESSAGE

    async def get_intermediate_step_parent_id(self, data_model: AIQResponseIntermediateStep) -> str:
        """
        Retrieves intermediate step parent_id from AIQResponseIntermediateStep instance.

        :param data_model: AIQResponseIntermediateStep Data Model instance.
        :return: Intermediate step parent_id or "default".
        """
        return data_model.parent_id or "root"

    async def create_system_response_token_message(  # pylint: disable=R0917:too-many-positional-arguments
        self,
        message_type: Literal[WebSocketMessageType.RESPONSE_MESSAGE,
                              WebSocketMessageType.ERROR_MESSAGE] = WebSocketMessageType.RESPONSE_MESSAGE,
        message_id: str | None = str(uuid.uuid4()),
        thread_id: str = "default",
        parent_id: str = "default",
        content: SystemResponseContent
        | Error = SystemResponseContent(),
        status: WebSocketMessageStatus = WebSocketMessageStatus.IN_PROGRESS,
        timestamp: str = str(datetime.datetime.now(datetime.timezone.utc))
    ) -> WebSocketSystemResponseTokenMessage | None:
        """
        Creates a system response token message with default values.

        :param message_type: Type of WebSocket message.
        :param message_id: Unique identifier for the message (default: generated UUID).
        :param thread_id: ID of the thread the message belongs to (default: "default").
        :param parent_id: ID of the user message that spawned child messages.
        :param content: Message content.
        :param status: Status of the message (default: IN_PROGRESS).
        :param timestamp: Timestamp of the message (default: current UTC time).
        :return: A WebSocketSystemResponseTokenMessage instance.
        """
        try:
            return WebSocketSystemResponseTokenMessage(type=message_type,
                                                       id=message_id,
                                                       thread_id=thread_id,
                                                       parent_id=parent_id,
                                                       content=content,
                                                       status=status,
                                                       timestamp=timestamp)

        except Exception as e:
            logger.error("Error creating system response token message: %s", str(e), exc_info=True)
            return None

    async def create_system_intermediate_step_message(  # pylint: disable=R0917:too-many-positional-arguments
        self,
        message_type: Literal[WebSocketMessageType.INTERMEDIATE_STEP_MESSAGE] = (
            WebSocketMessageType.INTERMEDIATE_STEP_MESSAGE),
        message_id: str = str(uuid.uuid4()),
        thread_id: str = "default",
        parent_id: str = "default",
        content: SystemIntermediateStepContent = SystemIntermediateStepContent(name="default", payload="default"),
        status: WebSocketMessageStatus = WebSocketMessageStatus.IN_PROGRESS,
        timestamp: str = str(datetime.datetime.now(datetime.timezone.utc))
    ) -> WebSocketSystemIntermediateStepMessage | None:
        """
        Creates a system intermediate step message with default values.

        :param message_type: Type of WebSocket message.
        :param message_id: Unique identifier for the message (default: generated UUID).
        :param thread_id: ID of the thread the message belongs to (default: "default").
        :param parent_id: ID of the user message that spawned child messages.
        :param content: Message content
        :param status: Status of the message (default: IN_PROGRESS).
        :param timestamp: Timestamp of the message (default: current UTC time).
        :return: A WebSocketSystemIntermediateStepMessage instance.
        """
        try:
            return WebSocketSystemIntermediateStepMessage(type=message_type,
                                                          id=message_id,
                                                          thread_id=thread_id,
                                                          parent_id=parent_id,
                                                          content=content,
                                                          status=status,
                                                          timestamp=timestamp)

        except Exception as e:
            logger.error("Error creating system intermediate step message: %s", str(e), exc_info=True)
            return None

    async def create_system_interaction_message(  # pylint: disable=R0917:too-many-positional-arguments
        self,
        *,
        message_type: Literal[WebSocketMessageType.SYSTEM_INTERACTION_MESSAGE] = (
            WebSocketMessageType.SYSTEM_INTERACTION_MESSAGE),
        message_id: str | None = str(uuid.uuid4()),
        thread_id: str = "default",
        parent_id: str = "default",
        content: HumanPrompt,
        status: WebSocketMessageStatus = WebSocketMessageStatus.IN_PROGRESS,
        timestamp: str = str(datetime.datetime.now(datetime.timezone.utc))
    ) -> WebSocketSystemInteractionMessage | None:  # noqa: E125 continuation line with same indent as next logical line
        """
        Creates a system interaction message with default values.

        :param message_type: Type of WebSocket message.
        :param message_id: Unique identifier for the message (default: generated UUID).
        :param thread_id: ID of the thread the message belongs to (default: "default").
        :param parent_id: ID of the user message that spawned child messages.
        :param content: Message content
        :param status: Status of the message (default: IN_PROGRESS).
        :param timestamp: Timestamp of the message (default: current UTC time).
        :return: A WebSocketSystemInteractionMessage instance.
        """
        try:
            return WebSocketSystemInteractionMessage(type=message_type,
                                                     id=message_id,
                                                     thread_id=thread_id,
                                                     parent_id=parent_id,
                                                     content=content,
                                                     status=status,
                                                     timestamp=timestamp)

        except Exception as e:
            logger.error("Error creating system interaction message: %s", str(e), exc_info=True)
            return None
