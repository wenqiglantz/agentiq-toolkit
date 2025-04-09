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

import typing
from abc import ABC
from abc import abstractmethod

from aiq.data_models.front_end import FrontEndConfigT

if (typing.TYPE_CHECKING):
    from aiq.data_models.config import AIQConfig


class FrontEndBase(typing.Generic[FrontEndConfigT], ABC):

    def __init__(self, full_config: "AIQConfig"):
        """
        Initializes the FrontEndBase object with the specified AgentIQ configuration.

        Parameters
        ----------
        full_config : AIQConfig
            The configuration object to use for the front end.
        """

        super().__init__()

        self._full_config: "AIQConfig" = full_config
        self._front_end_config: FrontEndConfigT = typing.cast(FrontEndConfigT, full_config.general.front_end)

    @property
    def front_end_config(self) -> FrontEndConfigT:
        """
        Returns the front end configuration object extracted from the AgentIQ configuration.

        Returns
        -------
        FrontEndConfigT
            The front end configuration object.
        """
        return self._front_end_config

    @property
    def full_config(self) -> "AIQConfig":
        """
        Returns the full AgentIQ configuration object.

        Returns
        -------
        AIQConfig
            The full AgentIQ configuration object.
        """

        return self._full_config

    @abstractmethod
    async def run(self):
        """
        Runs the specified configuration file, launching the workflow until the front end is complete.
        """
        pass
