#!/bin/bash
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

set -e

GITLAB_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source ${GITLAB_SCRIPT_DIR}/common.sh

create_env group:dev group:docs extra:examples

# Before running the checks, make sure we have no changes in the repo
git reset --hard

export PRE_COMMIT_HOME=${CI_PROJECT_DIR}/.cache/pre-commit

rapids-logger "Running checks"
${SCRIPT_DIR}/checks.sh

rapids-logger "Checking copyright headers"
python ${SCRIPT_DIR}/copyright.py --verify-apache-v2


rapids-logger "Runing Documentation checks"
${SCRIPT_DIR}/documentation_checks.sh
