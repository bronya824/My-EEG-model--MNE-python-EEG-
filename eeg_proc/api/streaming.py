# eeg_proc/api/streaming.py
import asyncio
import websockets
import json
import mne
import numpy as np
from ..processing.preprocessor import MNEPipeline
from concurrent.futures import ThreadPoolExecutor

class WebSocketAdapter:
    """合并后的 WebSocket 适配器（支持异步处理和线程池）"""
    def __init__(self, host='0.0.0.0', port=8765, max_workers=4):
        self.host = host
        self.port = port
        self._server = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)  # 新增线程池

    async def _handler(self, websocket):
        async for message in websocket:
            try:
                data = json.loads(message)
                raw = self._create_raw(
                    np.array(data['samples']),
                    data['sfreq'],
                    data['channels']
                )
                loop = asyncio.get_running_loop()
                # 使用线程池执行 CPU 密集型任务
                processed = await loop.run_in_executor(
                    self.executor,
                    lambda: MNEPipeline(raw).apply_standard_pipeline()
                )
                await websocket.send(json.dumps({
                    'status': 'success',
                    'data': processed.raw.get_data().tolist(),
                    'metadata': {
                        'sfreq': processed.raw.info['sfreq'],
                        'channels': processed.raw.ch_names
                    }
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': str(e)
                }))

    def _create_raw(self, data, sfreq, ch_names):
        return mne.io.RawArray(
            data,
            mne.create_info(ch_names, sfreq, 'eeg')
        )

    def start(self):
        """启动服务（兼容同步调用）"""
        async def _run_forever():
            async with websockets.serve(self._handler, self.host, self.port):
                await asyncio.Future()  # 永久运行

        asyncio.run(_run_forever())
if __name__ == '__main__':
    adapter = WebSocketAdapter()
    adapter.start()