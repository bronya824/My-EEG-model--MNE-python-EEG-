# eeg_proc/api/__init__.py
from flask import Flask

app = Flask(__name__)

__all__ = ['app']

def _init_components():
    global WebSocketAdapter, process_wechat, AliyunUploader, handle_edf_upload

    from .streaming import WebSocketAdapter
    __all__.append('WebSocketAdapter')

    from .rest_api import (
        process_wechat,
        AliyunUploader,
        handle_edf_upload
    )
    __all__.extend(['process_wechat', 'AliyunUploader', 'handle_edf_upload'])

def _register_routes():
    from .rest_api import handle_edf_upload

    app.add_url_rule('/api/v1/upload',
                     'handle_edf_upload',
                     handle_edf_upload,
                     methods=['POST'])

def __getattr__(name):
    if name == 'AliyunOutputModel':
        from ..models.eeg_model import AliyunOutputModel
        return AliyunOutputModel
    if name == 'ErrorResponse':
        from ..models.result_schema import ErrorResponse
        return ErrorResponse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_init_components()
_register_routes()

__all__ += [
    'AliyunOutputModel',
    'ErrorResponse'
]