# EEG Processor API Reference

## REST API Endpoints

### POST /process
处理EEG数据的核心端点

**请求格式**:
```json
{
  "device_id": "eeg-device-123",
  "user_id": "usr-20230918",
  "timestamp": "ISO8601时间戳",
  "sampling_rate": 500.0,
  "n_channels": 32,
  "channel_names": ["Fp1", "Fp2"],
  "eeg_data": "Base64编码的原始数据"
}