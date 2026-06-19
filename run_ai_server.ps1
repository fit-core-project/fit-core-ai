$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $AppDir

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$OllamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
$Ollama = if ($OllamaCommand) {
    $OllamaCommand.Source
} else {
    Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
}

if (-not (Test-Path $Ollama)) {
    throw "ollama.exe was not found. Install Ollama or add it to PATH."
}

$OllamaBaseUrl = if ($env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL } else { "http://127.0.0.1:11434" }
$LocalModel = if ($env:LOCAL_LLM_MODEL) { $env:LOCAL_LLM_MODEL } else { "gemma4:latest" }

try {
    $null = Invoke-RestMethod -Uri "$OllamaBaseUrl/api/version" -TimeoutSec 3
} catch {
    Start-Process -FilePath $Ollama -ArgumentList "serve" -WindowStyle Minimized

    $deadline = (Get-Date).AddSeconds(45)
    do {
        Start-Sleep -Seconds 2
        try {
            $null = Invoke-RestMethod -Uri "$OllamaBaseUrl/api/version" -TimeoutSec 3
            break
        } catch {
            if ((Get-Date) -ge $deadline) {
                throw "Ollama did not become ready at $OllamaBaseUrl within 45 seconds."
            }
        }
    } while ($true)
}

Start-Process -FilePath $Ollama -ArgumentList @("run", $LocalModel, ".") -WindowStyle Minimized

$VenvPython = Join-Path $AppDir "venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

& $Python -m uvicorn main:app --host 0.0.0.0 --port 8000
