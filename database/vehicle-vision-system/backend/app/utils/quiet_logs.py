"""抑制 MediaPipe / TensorFlow Lite 启动时的 C++ 警告输出。"""

from __future__ import annotations

import contextlib
import logging
import os
import sys


def configure_quiet_logs() -> None:
    """在导入 mediapipe 之前调用。"""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["GLOG_logtostderr"] = "0"
    os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
        absl.logging.set_stderrthreshold("error")
    except ImportError:
        pass

    for name in ("mediapipe", "tensorflow", "absl"):
        logging.getLogger(name).setLevel(logging.ERROR)


# Uvicorn 默认写 stderr，IDE 会把 stderr 整段标红；改到 stdout 并关闭红色等级着色
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(message)s",
            "use_colors": False,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            "use_colors": False,
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}


@contextlib.contextmanager
def suppress_native_stderr():
    """临时屏蔽 fd=2（C++ 库直接写 stderr 的 W0000 警告）。"""
    stderr_fd = sys.stderr.fileno()
    saved = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)
        os.close(devnull)
