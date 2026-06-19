$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AiScript = Join-Path $AppDir "run_ai_server.ps1"
$Model = if ($env:LOCAL_LLM_MODEL) { $env:LOCAL_LLM_MODEL } else { "gemma4:latest" }
$OllamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
$Ollama = if ($OllamaCommand) {
    $OllamaCommand.Source
} else {
    Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
}

$OllamaPane = "& `"$Ollama`" run $Model"
$AiPane = "powershell -NoExit -ExecutionPolicy Bypass -File `"$AiScript`""

if (Get-Command wt -ErrorAction SilentlyContinue) {
    wt -d $AppDir powershell -NoExit -Command $OllamaPane `; split-pane -H -d $AppDir $AiPane
} else {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $OllamaPane -WorkingDirectory $AppDir
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $AiScript -WorkingDirectory $AppDir
}
