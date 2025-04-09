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

import click

logger = logging.getLogger(__name__)


@click.group(name=__name__, invoke_without_command=True, help="Utility to add an AgentIQ remote registry channel.")
@click.argument("channel_type", type=str)
def add(channel_type: str) -> None:
    from aiq.utils.settings.global_settings import add_channel_interative

    add_channel_interative(channel_type=channel_type)
