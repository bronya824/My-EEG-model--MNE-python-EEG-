# eeg_proc/models/__init__.py
from .eeg_model import WechatEEGInput, AliyunOutputModel
from .result_schema import ErrorResponse, ProcessingResult

__all__ = [
    'WechatEEGInput',
    'AliyunOutputModel',
    'ErrorResponse',
    'ProcessingResult'
]