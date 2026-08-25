[CmdletBinding()]
param(
    [string]$Binary = $(if ($env:LLAMA_SERVER_BIN) { $env:LLAMA_SERVER_BIN } else { "llama-server.exe" }),
    [string]$ModelPath = $env:LLAMA_MODEL_PATH,
    [string]$HfRepo = $(if ($env:LLAMA_HF_REPO) { $env:LLAMA_HF_REPO } else { "unsloth/Qwen3.5-9B-GGUF:Q4_K_M" }),
    [string]$ModelAlias = $(if ($env:LLAMA_MODEL_ALIAS) { $env:LLAMA_MODEL_ALIAS } else { "qwen3.5-9b" }),
    [int]$Port = $(if ($env:LLAMA_PORT) { [int]$env:LLAMA_PORT } else { 18080 }),
    [int]$ContextSize = $(if ($env:LLAMA_CONTEXT_SIZE) { [int]$env:LLAMA_CONTEXT_SIZE } else { 32768 }),
    [int]$GpuLayers = $(if ($env:LLAMA_GPU_LAYERS) { [int]$env:LLAMA_GPU_LAYERS } else { 99 }),
    [int]$Parallel = $(if ($env:LLAMA_PARALLEL) { [int]$env:LLAMA_PARALLEL } else { 1 })
)

$resolvedBinary = $null
if (Test-Path -LiteralPath $Binary -PathType Leaf) {
    $resolvedBinary = (Resolve-Path -LiteralPath $Binary).Path
} else {
    $command = Get-Command $Binary -ErrorAction SilentlyContinue
    if ($command) {
        $resolvedBinary = $command.Source
    }
}

if (-not $resolvedBinary) {
    throw "未找到 llama-server。请安装 llama.cpp，或通过 -Binary / LLAMA_SERVER_BIN 指定 llama-server.exe。"
}

$serverArguments = @(
    "--alias", $ModelAlias,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--ctx-size", "$ContextSize",
    "--parallel", "$Parallel",
    "--n-gpu-layers", "$GpuLayers",
    "--jinja",
    "--no-mmproj"
)

if ($ModelPath) {
    if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
        throw "GGUF 模型不存在：$ModelPath"
    }
    $serverArguments = @("--model", (Resolve-Path -LiteralPath $ModelPath).Path) + $serverArguments
} else {
    $serverArguments = @("--hf-repo", $HfRepo) + $serverArguments
}

Write-Host "正在启动 llama.cpp：$ModelAlias -> http://127.0.0.1:$Port/v1"
Write-Host "上下文：$ContextSize，GPU layers：$GpuLayers，并发：$Parallel"
& $resolvedBinary @serverArguments
exit $LASTEXITCODE
