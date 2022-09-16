# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from . import _extension
from . import compliance
from . import datasets
from . import features
from . import functional
from . import io
from . import metric
from . import utils
from . import sox_effects
from . import streamdata
from . import text
from . import transform
from paddlespeech.audio.backends import get_audio_backend
from paddlespeech.audio.backends import list_audio_backends
from paddlespeech.audio.backends import set_audio_backend
from paddlespeech.audio.backends import soundfile_backend

__all__ = [
    "io",
    "compliance",
    "datasets",
    "functional",
    "features",
    "utils",
    "sox_effects",
    "streamdata",
    "text",
    "transform",
    "list_audio_backends",
    "get_audio_backend",
    "set_audio_backend",
    "soundfile_backend",
]
