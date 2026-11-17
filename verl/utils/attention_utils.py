# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import inspect
from dataclasses import dataclass
from typing import Callable, Optional

_index_first_axis, _pad_input, _rearrange, _unpad_input = None, None, None, None


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    """Dynamically import attention functions based on available hardware."""

    from verl.utils.device import is_torch_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input

    # flash-attn only ships CUDA wheels, so every other backend -- NPU, and any
    # device without a flash-attn build -- as well as CPU-only installs (unit
    # tests, dev boxes) use the pure-torch port in attention_padding_utils.
    use_flash_attn = not is_torch_npu_available(check_device=False)
    if use_flash_attn:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ImportError:
            use_flash_attn = False

    if not use_flash_attn:
        from verl.utils.attention_padding_utils import index_first_axis, pad_input, rearrange, unpad_input

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input

    return _index_first_axis, _pad_input, _rearrange, _unpad_input


def index_first_axis(*args, **kwargs):
    """
    Unified entry point for `index_first_axis` across backends.

    Dynamically dispatches to the appropriate implementation:
      - With flash-attn installed: `flash_attn.bert_padding.index_first_axis`
      - Otherwise: `verl.utils.attention_padding_utils.index_first_axis`

    Users can call this function directly without worrying about the underlying device.
    """
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args, **kwargs):
    """
    Unified entry point for `pad_input` across backends.

    Dynamically dispatches to the appropriate implementation:
      - With flash-attn installed: `flash_attn.bert_padding.pad_input`
      - Otherwise: `verl.utils.attention_padding_utils.pad_input`

    Users can call this function directly without worrying about the underlying device.
    """
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args, **kwargs):
    """
    Unified entry point for `rearrange` across backends.

    Dynamically dispatches to the appropriate implementation:
      - With flash-attn installed: `flash_attn.bert_padding.rearrange`
      - Otherwise: `einops.rearrange`, re-exported from
        `verl.utils.attention_padding_utils`

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args, **kwargs):
    """
    Unified entry point for `unpad_input` across backends.

    Dynamically dispatches to the appropriate implementation:
      - With flash-attn installed: `flash_attn.bert_padding.unpad_input`
      - Otherwise: `verl.utils.attention_padding_utils.unpad_input`

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


@dataclass(frozen=True)
class FlashAttnFunctions:
    """FlashAttention-2 varlen kernel for the current device, with the flags its callers need."""

    varlen_func: Callable
    supports_window_size: bool
    supports_deterministic: bool
    use_top_left_mask: bool


_flash_attn_functions: Optional[FlashAttnFunctions] = None


def get_flash_attn_functions() -> FlashAttnFunctions:
    """Return FlashAttention-2 for the current device, resolved on first call.

    - NPU: the transformers NPU flash-attention integration.
    - With flash-attn installed: the `flash_attn` package.
    - Intel GPU without flash-attn: the hub kernel loaded by transformers
      (`kernels-community/flash-attn2`), fetched on this first call.

    Raises:
        RuntimeError: If no FlashAttention-2 implementation exists for the device.
    """
    global _flash_attn_functions
    if _flash_attn_functions is not None:
        return _flash_attn_functions

    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
    from transformers.utils import (
        is_flash_attn_2_available,
        is_flash_attn_greater_or_equal_2_10,
        is_torch_xpu_available,
    )

    from verl.utils.device import is_npu_available

    if is_npu_available:
        from transformers.integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
        from transformers.integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_varlen_func

        use_top_left_mask = flash_attn_supports_top_left_mask()
    elif is_flash_attn_2_available():
        from flash_attn import flash_attn_func, flash_attn_varlen_func

        use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    elif is_torch_xpu_available():
        from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

        (flash_attn_func, flash_attn_varlen_func, *_), _ = lazy_import_flash_attention("flash_attention_2")
        use_top_left_mask = flash_attn_supports_top_left_mask()
    else:
        raise RuntimeError("FlashAttention-2 is required here: install flash-attn, or run on NPU or Intel GPU.")

    params = inspect.signature(flash_attn_func).parameters
    _flash_attn_functions = FlashAttnFunctions(
        varlen_func=flash_attn_varlen_func,
        supports_window_size="window_size" in params,
        supports_deterministic="deterministic" in params,
        use_top_left_mask=use_top_left_mask,
    )
    return _flash_attn_functions


__all__ = [
    "FlashAttnFunctions",
    "get_flash_attn_functions",
    "index_first_axis",
    "pad_input",
    "rearrange",
    "unpad_input",
]
