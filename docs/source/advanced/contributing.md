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

# Contributing to NVIDIA AgentIQ

Contributions to AgentIQ fall into the following three categories.

1. To report a bug, request a new feature, or report a problem with
    documentation, file a [bug](https://github.com/NVIDIA/AgentIQ/issues)
    describing in detail the problem or new feature. The AgentIQ team evaluates
    and triages bugs and schedules them for a release. If you believe the
    bug needs priority attention, comment on the bug to notify the
    team.
2. To propose and implement a new Feature, file a new feature request
    [issue](https://github.com/NVIDIA/AgentIQ/issues). Describe the
    intended feature and discuss the design and implementation with the team and
    community. Once the team agrees that the plan is good, go ahead and
    implement it, using the [code contributions](#code-contributions) guide below.
3. To implement a feature or bug-fix for an existing outstanding issue,
    follow the [code contributions](#code-contributions) guide below. If you
    need more context on a particular issue, ask in a comment.

As contributors and maintainers of AgentIQ, you are expected to abide by AgentIQ's code of conduct. More information can be found at: [Contributor Code of Conduct](../code-of-conduct.md).

## Set Up Your Development Environment
### Prerequisites

- Install [Git](https://git-scm.com/)
- Install [Git Large File Storage](https://git-lfs.github.com/) (LFS)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Install [Visual Studio Code](https://code.visualstudio.com/) (recommended)

AgentIQ is a Python library that doesn’t require a GPU to run the workflow by default. You can deploy the core workflows using one of the following:
- Ubuntu or other Linux distributions, including WSL, in a Python virtual environment.

### Creating the Environment

1. Fork the AgentIQ repository choosing **Fork** on the [AgentIQ repository page](https://github.com/NVIDIA/AgentIQ).

1. Clone your personal fork of the AgentIQ repository to your local machine.
    ```bash
    git clone <your fork url> agentiq
    cd agentiq
    ```

    Then, set the upstream to the main repository and fetch the latest changes:
    ```bash
    git remote add upstream git@github.com:NVIDIA/AgentIQ.git
    git fetch --all
    ```


1. Initialize, fetch, and update submodules in the Git repository.
    ```bash
    git submodule update --init --recursive
    ```

1. Fetch the data sets by downloading the LFS files.
    ```bash
    git lfs install
    git lfs fetch
    git lfs pull
    ```

1. Create a Python environment.
    ```bash
    uv venv --seed .venv
    source .venv/bin/activate
    uv sync --all-groups --all-extras
    ```

1. Install and configure pre-commit hooks.

    ```bash
    pre-commit install
    ```
    **NOTE**: Running pre-commit for the first time will take longer than normal.

7. Open the AgentIQ Workspace in Visual Studio Code.
    ```bash
    code ./aiq.code-workspace
    ```

### Install the AgentIQ Library

1. Install the AgentIQ Examples by doing the following.
   - Install AgentIQ examples.

     ```bash
     uv sync --extra examples
     ```
   - Install a single example by running `uv pip install -e ./examples/<example_name>`.
   For example, install the Simple Calculator example with the following command.

     ```bash
     uv pip install -e ./examples/simple_calculator
     ```

2. Verify that you've installed the AgentIQ library.

     ```bash
     aiq --help
     aiq --version
     ```

     If the installation succeeded, the `aiq` command will log the help message and its current version.


## Code contributions

### Your first issue

1. Find an issue to work on. The best way is to search for issues with the [good first issue](https://github.com/NVIDIA/AgentIQ/issues) label.
1. Make sure that you can contribute your work to open source (no license and/or patent conflict is introduced by your code). You will need to [`sign`](#signing-your-work) your commit.
1. Comment on the issue stating that you are going to work on it.
1. [Fork the AgentIQ repository](https://github.com/NVIDIA/AgentIQ/fork)
1. Code!
    - Make sure to update unit tests!
    - Ensure the [license headers are set properly](#licensing).
1. Verify your changes by [running CI locally](./running-ci-locally.md) with the `./ci/scripts/run_ci_local.sh all` command.
1. When done, [create your pull request](https://github.com/NVIDIA/AgentIQ/compare). Select `develop` as the `Target branch` of your pull request.
    - Ensure the body of the pull request references the issue you are working on in the form of `Closes #<issue number>`.
1. Wait for other developers to review your code and update code as needed.
1. Once reviewed and approved, an AgentIQ developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!

### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO is available at [Developer Certificate of Origin](https://developercertificate.org/)

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you can review the prioritized issues for our next release in our [project boards](https://github.com/NVIDIA/AgentIQ/projects).

> **Pro Tip:** Always review the release board with the highest number for issues to work on. This is where AgentIQ developers also focus their efforts.

Review the unassigned issues and choose an issue that you are comfortable contributing. Ensure you comment on the issue before you begin to inform others that you are working on it. If you have questions about implementing the issue, comment your questions in the issue instead of the PR.

## Developing with AgentIQ

Refer to the [Get Started](../intro/get-started.md) guide to quickly begin development.


## Licensing
AgentIQ is licensed under the Apache v2.0 license. All new source files including CMake and other build scripts should contain the Apache v2.0 license header. Any edits to existing source code should update the date range of the copyright to the current year. The format for the license header is:

```python
# SPDX-FileCopyrightText: Copyright (c) <year>, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
```

### Third-party code
Third-party code included in the source tree (that is not pulled in as an external dependency) must be compatible with the Apache v2.0 license and should retain the original license along with a URL to the source. If this code is modified, it should contain both the Apache v2.0 license followed by the original license of the code and the URL to the original code.

Ex:
```python
# SPDX-FileCopyrightText: Copyright (c) <year>, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#
# Original Source: https://github.com/org/other_project
#
# Original License:
# ...
```
