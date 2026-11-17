"""Ask the operating system to treat this process as latency-sensitive.

On Apple silicon, macOS decides per thread whether to run on a performance or
an efficiency core, guided by the thread's quality-of-service class. A mostly
idle command-line process gets the default class and can be parked on
efficiency cores, which run slower. "User interactive" is the class macOS uses
for work a user is waiting on and prefers performance cores for it.

This is a request, not a guarantee, and it only exists on macOS. Elsewhere the
function does nothing and says so. Whether it helps is measured, not assumed:
`live.bench --idle-ms 1000 --qos default` against `--qos interactive`.
"""
from __future__ import annotations

import ctypes
import platform

QOS_CLASS_USER_INTERACTIVE = 0x21


def request_interactive_qos() -> str:
    """Set the calling thread's QoS class. Returns a short status for the log."""
    if platform.system() != "Darwin":
        return f"not applied: {platform.system()} has no QoS classes"
    try:
        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        fn = libsystem.pthread_set_qos_class_self_np
        fn.argtypes = [ctypes.c_uint, ctypes.c_int]
        fn.restype = ctypes.c_int
        rc = fn(QOS_CLASS_USER_INTERACTIVE, 0)
    except (OSError, AttributeError) as exc:
        return f"not applied: {exc}"
    return "user-interactive" if rc == 0 else f"not applied: error {rc}"
