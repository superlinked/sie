# Copyright 2024 Superlinked, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from superlinked.framework.common.space.embedding.model_based.model_dimension_cache import (
    MODEL_DIMENSION_BY_NAME,
)


class TestModelDimensionCache:
    def test_embo01_dimension_cached(self) -> None:
        assert "embo-01" in MODEL_DIMENSION_BY_NAME
        assert MODEL_DIMENSION_BY_NAME["embo-01"] == 1536
