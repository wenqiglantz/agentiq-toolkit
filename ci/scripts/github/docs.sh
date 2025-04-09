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

GITHUB_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source ${GITHUB_SCRIPT_DIR}/common.sh

rapids-logger "Installing non-pip deps"
get_lfs_files

create_env group:dev group:docs

rapids-logger "Building documentation"
pushd ${PROJECT_ROOT}/docs
make html

DOCS_TAR=${WORKSPACE_TMP}/docs.tar.bz2
rapids-logger "Archiving documentation to ${DOCS_TAR}"
tar cvfj ${DOCS_TAR} build/html
popd

rapids-logger "Documentation build completed"
