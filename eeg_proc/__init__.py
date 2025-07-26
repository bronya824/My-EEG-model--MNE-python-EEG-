# eeg_proc/__init__.py
"""
EEG处理器主模块 - 递归问题终极修复
"""

import sys
import importlib
from types import ModuleType
from typing import Any

__all__ = []  # 动态填充导出项


# --------------------- 安全延迟加载器 ---------------------
class _SafeLazyLoader(ModuleType):
    """防递归延迟加载器 (通过导入状态隔离)"""

    def __init__(self, name: str, original_module: ModuleType):
        super().__init__(name)
        self.__original_module = original_module
        self.__initialized = False
        self.__lock = False  # 线程安全锁（简单版）

    def __getattr__(self, name: str) -> Any:
        if self.__lock:
            # 防止在初始化过程中二次触发
            return super().__getattr__(name)

        if not self.__initialized:
            self.__lock = True
            try:
                self._init_modules()
            finally:
                self.__lock = False
                self.__initialized = True
        return getattr(self.__original_module, name)

    def _init_modules(self) -> None:
        """安全初始化核心组件"""
        # 临时恢复原始模块防止递归
        sys.modules[self.__name__] = self.__original_module

        try:
            # 显式导入所有依赖项
            from .processing.preprocessor import MNEPipeline
            from .io.receiver import EEGReceiver
            from .api.streaming import WebSocketAdapter
            from .models.eeg_model import (
                AliyunOutputModel,
                WechatEEGInput
            )

            # 注册全局导出项
            global __all__
            __all__ += [
                "MNEPipeline",
                "EEGReceiver",
                "WebSocketAdapter",
                "AliyunOutputModel",
                "WechatEEGInput"
            ]

            # 直接注入原始模块
            self.__original_module.MNEPipeline = MNEPipeline
            self.__original_module.EEGReceiver = EEGReceiver
            self.__original_module.WebSocketAdapter = WebSocketAdapter
            self.__original_module.AliyunOutputModel = AliyunOutputModel
            self.__original_module.WechatEEGInput = WechatEEGInput
        finally:
            # 恢复延迟加载器
            sys.modules[self.__name__] = self


# --------------------- 模块替换逻辑 ---------------------
original_module = sys.modules[__name__]
sys.modules[__name__] = _SafeLazyLoader(__name__, original_module)

# --------------------- 类型提示占位符 ---------------------
# 以下仅为IDE静态类型检查使用
MNEPipeline: type = None  # type: ignore
EEGReceiver: type = None  # type: ignore
WebSocketAdapter: type = None  # type: ignore
AliyunOutputModel: type = None  # type: ignore
WechatEEGInput: type = None  # type: ignore