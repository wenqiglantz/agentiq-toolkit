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
WHEELS_DIR=${CI_PROJECT_DIR}/.tmp/wheels

rapids-logger "Git Version: $(git describe)"

create_env extra:all


function get_git_tag() {
    # Get the latest Git tag, sorted by version, excluding lightweight tags
    git describe --tags --abbrev=0 2>/dev/null || echo "no-tag"
}
GIT_TAG=$(get_git_tag)


function build_wheel() {
    rapids-logger "Building Wheel for $1"
    uv build --wheel --no-progress --out-dir "${WHEELS_DIR}/$2" --directory $1
}

build_wheel . "agentiq/${GIT_TAG}"


# Build all examples with a pyproject.toml in the first directory below examples
for AIQ_EXAMPLE in ${AIQ_EXAMPLES[@]}; do
    # places all wheels flat under example
    build_wheel ${AIQ_EXAMPLE} "examples"
done

# Build all packages with a pyproject.toml in the first directory below packages
for AIQ_PACKAGE in "${AIQ_PACKAGES[@]}"; do
    # drop each package into a separate directory
    PACKAGE_DIR_NAME="${AIQ_PACKAGE#packages/}"
    PACKAGE_DIR_NAME="${AIQ_PACKAGE#./packages/}"
    # Replace "aiq_" with "agentiq_"
    PACKAGE_DIR_NAME="${PACKAGE_DIR_NAME//aiq_/agentiq_}"
    build_wheel "${AIQ_PACKAGE}" "${PACKAGE_DIR_NAME}/${GIT_TAG}"
done

if [[ "${CI_COMMIT_BRANCH}" == "${CI_DEFAULT_BRANCH}" || "${CI_COMMIT_BRANCH}" == "main" ]]; then
    rapids-logger "Uploading Wheels"

    # Find and upload all .whl files from nested directories
    while read -r WHEEL_FILE; do
        echo "Uploading ${WHEEL_FILE}..."

        python -m twine upload \
            -u gitlab-ci-token \
            -p "${CI_JOB_TOKEN}" \
            --non-interactive \
            --repository-url "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/pypi" \
            "${WHEEL_FILE}"
    done < <(find "${WHEELS_DIR}" -type f -name "*.whl")
fi
