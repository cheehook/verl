# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""An external module that detects the platform at import time sees plugin platforms."""

import os
import subprocess
import sys
import textwrap

_PLUGIN = textwrap.dedent(
    """
    from verl.plugin.platform.platform_cuda import PlatformCUDA
    from verl.plugin.platform.platform_manager import PlatformRegistry


    @PlatformRegistry.register("fakeplat")
    class FakePlatform(PlatformCUDA):
        @property
        def device_name(self):
            return "fakedev"

        def is_available(self):
            return True

        def is_platform_available(self, use_smi_check=False):
            return True
    """
)

# Mirrors an external module (e.g. one importing verl's FSDP engine) that reads the
# device name at module scope.
_EXTERNAL = textwrap.dedent(
    """
    from verl.utils.device import get_device_name

    DEVICE_AT_IMPORT = get_device_name()
    """
)


def test_external_module_sees_plugin_platform(tmp_path):
    (tmp_path / "fake_verl_plugin.py").write_text(_PLUGIN)
    (tmp_path / "fake_external_module.py").write_text(_EXTERNAL)
    dist_info = tmp_path / "fake_verl_plugin-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-verl-plugin\nVersion: 0.0.0\n")
    (dist_info / "entry_points.txt").write_text("[verl.plugins]\nfake = fake_verl_plugin\n")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    env["VERL_PLATFORM"] = "fakeplat"
    env["VERL_USE_EXTERNAL_MODULES"] = "fake_external_module"
    env["VERL_USE_EXTERNAL_PLUGINS"] = "fake"
    code = "import verl, fake_external_module; print(fake_external_module.DEVICE_AT_IMPORT)"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "fakedev"
