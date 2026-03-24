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

from beartype.typing import Mapping

MODEL_DIMENSION_BY_NAME: Mapping[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "Alibaba-NLP/gte-large-en-v1.5": 1024,
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": 1536,
    "clip-ViT-B-32": 512,
    "RN50": 1024,
    "Marqo/marqo-fashionSigLIP": 768,
    "pySilver/marqo-fashionSigLIP-ST": 768,
    "embo-01": 1536,
}
