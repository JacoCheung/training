# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict

import unittest

try:
    import torch
    from torchrec.distributed.embedding import EmbeddingCollectionSharder

    from generative_recommenders.dlrm_v3.train.utils import (
        _maybe_apply_qcomm_a2a,
    )

    _HAVE_DEPS = True
except Exception:  # pragma: no cover - import guard for environments without TorchRec
    _HAVE_DEPS = False


@unittest.skipUnless(_HAVE_DEPS, "torch / torchrec not importable")
class EmbeddingDedupConfigTest(unittest.TestCase):
    def test_default_disabled_path_preserves_sharder(self) -> None:
        sharder = EmbeddingCollectionSharder()

        configured = _maybe_apply_qcomm_a2a(
            [sharder],
            torch.device("cpu"),
            use_index_dedup=False,
        )

        self.assertIs(configured[0], sharder)
        self.assertFalse(configured[0]._use_index_dedup)

    def test_enabled_path_replaces_sharder_with_dedup_enabled(self) -> None:
        other_sharder = object()
        original = EmbeddingCollectionSharder()

        configured = _maybe_apply_qcomm_a2a(
            [other_sharder, original],
            torch.device("cpu"),
            use_index_dedup=True,
        )

        self.assertIs(configured[0], other_sharder)
        self.assertIsNot(configured[1], original)
        self.assertTrue(configured[1]._use_index_dedup)


if __name__ == "__main__":
    unittest.main()
