from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import sys
import types


def ensure_torch_six():
    if "torch._six" in sys.modules:
        return
    six_module = types.ModuleType("torch._six")
    six_module.string_classes = (str, bytes)
    six_module.int_classes = (int,)
    sys.modules["torch._six"] = six_module
