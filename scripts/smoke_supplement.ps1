$ErrorActionPreference = "Stop"

param(
    [string]$BaseUrl = "http://100.114.8.118:8000"
)

Invoke-RestMethod -Uri "$BaseUrl/api/ai/health" -Method Get | ConvertTo-Json -Depth 6

$Questions = @(
    "종합비타민, 오메가3, 실리마린을 한번에 섭취해도 괜찮아?",
    "유산균 항생제랑 같이 먹어도 돼?",
    "아연이랑 철분 같이 먹어도 돼?",
    "신장질환 있는데 프로틴 먹어도 돼?",
    "임산부 종합비타민 먹어도 돼?"
)

foreach ($Question in $Questions) {
    $Body = @{ question = $Question } | ConvertTo-Json -Compress
    Write-Output "QUESTION: $Question"
    Invoke-RestMethod `
        -Uri "$BaseUrl/api/ai/supplement-chat" `
        -Method Post `
        -ContentType "application/json; charset=utf-8" `
        -Body $Body |
        ConvertTo-Json -Depth 8
}
