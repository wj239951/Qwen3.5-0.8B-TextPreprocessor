# 一键复现：规则数据 -> 老师数据 -> SFT -> 合并 -> 评测 -> 模型卡
param(
    [int]$TrainPairs = 12000,
    [int]$TeacherPairs = 1200,
    [int]$TeacherRepeat = 2,
    [switch]$SkipTeacher,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = if (Test-Path "$root\.venv\Scripts\python.exe") { "$root\.venv\Scripts\python.exe" } else { "python" }
$env:PYTHONIOENCODING = "utf-8"

function Step($name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }

Step "1/6 规则合成数据"
& $py src/data/build_rule_pairs.py --train $TrainPairs --valid 600 --test 600

if (-not $SkipTeacher) {
    Step "2/6 老师生成数据"
    if (-not $env:DEEPSEEK_API_KEY) { throw "需要设置 DEEPSEEK_API_KEY" }
    & $py src/data/teacher_pairs.py --mode gen --n $TeacherPairs --workers 10
    & $py tools/relabel_teacher.py --in data/teacher/gen.jsonl --out data/teacher/gen_v2.jsonl
}

Step "3/6 合并训练集"
& $py src/data/build_sft.py --teacher data/teacher/gen_v2.jsonl --teacher-repeat $TeacherRepeat

if (-not $SkipTraining) {
    Step "4/6 LoRA 训练"
    & $py src/train/sft_lora.py --data data/sft/train.jsonl --valid data/sft/valid.jsonl `
        --out outputs/lora --epochs 2 --batch-size 1 --grad-accum 16 --max-len 512

    Step "5/6 合并权重"
    & $py src/train/merge_lora.py --base models/Qwen3.5-0.8B --adapter outputs/lora `
        --out Qwen3.5-0.8B-TextPreprocessor
}

Step "6/6 评测 + 模型卡"
& $py src/eval/eval_model.py --model Qwen3.5-0.8B-TextPreprocessor `
    --data data/rule/test.jsonl --baseline --out eval/rule_test.json
& $py tools/make_model_card.py --release Qwen3.5-0.8B-TextPreprocessor `
    --eval eval/rule_test.json

Write-Host "`n完成。" -ForegroundColor Green
