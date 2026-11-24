# Copy this file to local-config.ps1 and edit only when customization is needed.
$env:BGM_HOST = "127.0.0.1"
$env:BGM_PORT = "8765"

# $env:BGM_DATA_DIR = "D:\BgmLibraryData"
# $env:BGM_CONDA_ENV = "bgm-v2"
# $env:BGM_FFMPEG = "D:\tools\ffmpeg\bin\ffmpeg.exe"
# $env:BGM_FFPROBE = "D:\tools\ffmpeg\bin\ffprobe.exe"
# $env:BGM_CLAP_MODEL = "D:\models\clap-htsat-unfused"
# $env:BGM_ANALYSIS_MODEL = "D:\models\ast-finetuned-audioset"
# $env:BGM_TEXT_MODEL = "D:\models\bge-small-zh-v1.5"

# AST batch size. Use 2 for a 4 GB NVIDIA GPU; lower to 1 if other apps use VRAM.
# $env:BGM_AST_BATCH_SIZE = "2"

# Recommended-tag classifier. "local" is fully offline and is the default.
$env:BGM_RECOMMENDED_PROVIDER = "local"

# To use Alibaba Cloud Model Studio, change provider to "aliyun" and configure:
# $env:BGM_RECOMMENDED_PROVIDER = "aliyun"
# $env:DASHSCOPE_API_KEY = "sk-your-key"
# $env:DASHSCOPE_BASE_URL = "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
# $env:BGM_ALIYUN_MODEL = "qwen3.8-omni-flash"
# $env:BGM_CLOUD_MIN_CONFIDENCE = "0.78"
# $env:BGM_CLOUD_FALLBACK_LOCAL = "1"
# $env:BGM_CLOUD_TIMEOUT_SECONDS = "120"
# $env:BGM_CLOUD_MAX_CONCURRENCY = "2"
# $env:BGM_CLOUD_UCS_CATEGORY_CONFIDENCE = "0.85"
# $env:BGM_CLOUD_UCS_SUBCATEGORY_CONFIDENCE = "0.82"
