# 将本文件复制为“本地配置.ps1”后按需修改。全部配置均为可选。

# 服务地址：默认仅本机可以访问。
$env:BGM_HOST = "127.0.0.1"
$env:BGM_PORT = "8765"

# 数据库位置。默认是项目目录下的 data 文件夹。
# $env:BGM_DATA_DIR = "D:\BgmLibraryData"

# Conda 环境名。默认是 bgm-v2。
# $env:BGM_CONDA_ENV = "bgm-v2"

# 仅当 ffmpeg 没有安装进 Conda 环境时才需要指定。
# $env:BGM_FFMPEG = "D:\tools\ffmpeg\bin\ffmpeg.exe"
# $env:BGM_FFPROBE = "D:\tools\ffmpeg\bin\ffprobe.exe"

# AST 推理批次。4GB NVIDIA 显卡建议为 2；显存被其他程序占用时可设为 1。
# $env:BGM_AST_BATCH_SIZE = "2"

# 推荐标签分类器。local 为完全离线，也是默认值。
$env:BGM_RECOMMENDED_PROVIDER = "local"

# 使用阿里云百炼时，改为 aliyun 并填写同一地域的 API Key 和业务空间地址。
# $env:BGM_RECOMMENDED_PROVIDER = "aliyun"
# $env:DASHSCOPE_API_KEY = "sk-请替换"
# $env:DASHSCOPE_BASE_URL = "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
# $env:BGM_ALIYUN_MODEL = "qwen3.8-omni-flash"
# $env:BGM_CLOUD_MIN_CONFIDENCE = "0.78"
# $env:BGM_CLOUD_FALLBACK_LOCAL = "1"
# $env:BGM_CLOUD_TIMEOUT_SECONDS = "120"
# $env:BGM_CLOUD_MAX_CONCURRENCY = "2"
# $env:BGM_CLOUD_UCS_CATEGORY_CONFIDENCE = "0.85"
# $env:BGM_CLOUD_UCS_SUBCATEGORY_CONFIDENCE = "0.82"

# 使用已经离线下载的 CLAP 模型目录，可避免首次运行联网下载。
# $env:BGM_CLAP_MODEL = "D:\models\clap-htsat-unfused"
