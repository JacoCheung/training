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

from generative_recommenders.dlrm_v3.configs import get_hstu_configs


class HSTUDropoutConfigTest(unittest.TestCase):
    def test_default_preserves_configured_ratios(self) -> None:
        config = get_hstu_configs(dataset="debug")

        self.assertEqual(config.hstu_input_dropout_ratio, 0.2)
        self.assertEqual(config.hstu_linear_dropout_rate, 0.1)

    def test_disabled_zeroes_all_configured_ratios(self) -> None:
        config = get_hstu_configs(
            dataset="debug",
            hstu_input_dropout_ratio=0.4,
            hstu_linear_dropout_rate=0.3,
            enable_dropout=0,
        )

        self.assertEqual(config.hstu_input_dropout_ratio, 0.0)
        self.assertEqual(config.hstu_linear_dropout_rate, 0.0)

    def test_invalid_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            get_hstu_configs(dataset="debug", enable_dropout=2)


if __name__ == "__main__":
    unittest.main()
