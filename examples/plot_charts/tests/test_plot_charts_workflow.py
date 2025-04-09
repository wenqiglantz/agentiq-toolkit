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

import importlib
import importlib.resources
import inspect
import logging
from pathlib import Path

import pytest
from aiq_plot_charts.register import PlotChartsWorkflowConfig

from aiq.runtime.loader import load_workflow

logger = logging.getLogger(__name__)


@pytest.mark.e2e
async def test_full_workflow():

    package_name = inspect.getmodule(PlotChartsWorkflowConfig).__package__

    config_file: Path = importlib.resources.files(package_name).joinpath("configs", "config.yml").absolute()

    async with load_workflow(config_file) as workflow:

        async with workflow.run("make a line chart for me") as runner:

            result = await runner.result(to_type=str)

        result = result.lower()
        assert "saved output to" in result
