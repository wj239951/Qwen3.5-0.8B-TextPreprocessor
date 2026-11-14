# 训练完成后的收尾：合并 -> 自检 -> 评测 -> 模型卡 -> 复制到桌面
param(
    [string]$Adapter = "outputs/lora",
    [string]$Out = "Qwen3.5-0.8B-TextPreprocessor",
    [switch]$SkipDesktop
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = if (Test-Path "$root\.venv\Scripts\python.exe") { "$root\.venv\Scripts\python.exe" } else { "python" }
$env:PYTHONIOENCODING = "utf-8"

function Step($name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }

Step "1/6 合并 LoRA"
& $py src/train/merge_lora.py --base models/Qwen3.5-0.8B --adapter $Adapter --out $Out

Step "2/6 成品自检"
& $py tools/verify_release.py --model $Out --limit 12

Step "3/6 评测（规则测试集）"
& $py src/eval/eval_model.py --model $Out --data data/rule/test.jsonl --baseline `
    --out eval/rule_test.json --save-preds eval/preds

Step "4/6 评测（老师测试集）"
& $py src/eval/eval_model.py --model $Out --data data/teacher/test_v3.jsonl --baseline `
    --out eval/teacher_test.json --save-preds eval/preds_teacher

Step "5/6 示例 + 模型卡"
& $py tools/make_examples.py --preds eval/preds_rule_test.jsonl --out examples.md
Copy-Item "examples.md" (Join-Path $Out "examples.md") -Force
& $py tools/make_model_card.py --release $Out `
    --eval eval/rule_test.json --eval-teacher eval/teacher_test.json `
    --train-meta $Adapter/train_meta.json

Step "6/6 复制到桌面"
if (-not $SkipDesktop) {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $target = Join-Path $desktop (Split-Path -Leaf $Out)
    if (Test-Path $target) { Remove-Item $target -Recurse -Force }
    Copy-Item $Out $target -Recurse
    $size = (Get-ChildItem $target -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
    Write-Host ("已复制到 {0}（{1:N2} GB）" -f $target, $size) -ForegroundColor Green
}

Write-Host "`n收尾完成。" -ForegroundColor Green
