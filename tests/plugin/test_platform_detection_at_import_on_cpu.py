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
"""Importing verl's engines must not detect the platform.

verl imports the modules in ``VERL_USE_EXTERNAL_MODULES`` and then loads the
``verl.plugins`` entry points. Either side may import the engines (an external
trainer package, or a hardware plugin registering its engines), and either side may
register a platform. If importing the engines detected the platform, whichever
registers its platform second would never be detected.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_REGISTER_PLATFORM = textwrap.dedent(
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

_IMPORT_ENGINES = "import verl.workers.engine  # noqa: F401\n"


def _run(code, env_overrides):
    env = dict(os.environ)
    env.pop("VERL_USE_EXTERNAL_MODULES", None)
    env.pop("VERL_PLATFORM", None)
    env.update(env_overrides)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)


def test_engine_import_does_not_detect_platform():
    code = (
        "import verl, verl.workers.engine\n"
        "from verl.plugin.platform import platform_manager\n"
        "print(platform_manager._current_platform is None)"
    )
    result = _run(code, {"VERL_USE_EXTERNAL_PLUGINS": "none"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "True"


@pytest.mark.parametrize(
    "plugin_src, external_src",
    [
        pytest.param(_REGISTER_PLATFORM, _IMPORT_ENGINES, id="plugin-registers-platform"),
        pytest.param(_IMPORT_ENGINES, _REGISTER_PLATFORM, id="external-module-registers-platform"),
    ],
)
def test_platform_registered_by_either_side_is_used(tmp_path, plugin_src, external_src):
    (tmp_path / "fake_verl_plugin.py").write_text(plugin_src)
    (tmp_path / "fake_external_module.py").write_text(external_src)
    dist_info = tmp_path / "fake_verl_plugin-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-verl-plugin\nVersion: 0.0.0\n")
    (dist_info / "entry_points.txt").write_text("[verl.plugins]\nfake = fake_verl_plugin\n")

    code = "import verl\nfrom verl.utils.device import get_device_name\nprint(get_device_name())"
    result = _run(
        code,
        {
            "PYTHONPATH": os.pathsep.join(filter(None, [str(tmp_path), os.environ.get("PYTHONPATH")])),
            "VERL_PLATFORM": "fakeplat",
            "VERL_USE_EXTERNAL_MODULES": "fake_external_module",
            "VERL_USE_EXTERNAL_PLUGINS": "fake",
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "fakedev"
