# 开源发布清单

发布前逐项打勾。每一项都对应一次真实的翻车风险。

## 1. 权重与目录

- [ ] 模型目录里 `config.json`、`model.safetensors*`、`tokenizer.json`、
      `tokenizer_config.json`、`chat_template.jinja` 齐全
- [ ] **没有** `adapter_config.json`（那是 LoRA 中间产物，不是可分发权重）
- [ ] **没有** 404/错误页伪装成的 json（下载小文件时最容易被 ModelScope 的错误页骗过，
      检查方法：`Get-Content config.json` 第一行应该是 `{` 而不是 `{"Code":...}`）
- [ ] `generation_config.json` 的 `eos_token_id` 是 `<|im_end|>`（248046）而不是
      `<|endoftext|>`（248044），否则用户 `model.generate()` 会停不下来
- [ ] `python tools/verify_release.py --model <dir>` 全绿
- [ ] 目录总大小合理（fp16 约 1.6GB）

## 2. 模型卡

- [ ] `README.md` 的 YAML front matter 有 `license`、`base_model`、`language`、`tags`
- [ ] 能力表和限制表都写了"不做什么"（尤其是"不修乱码"）
- [ ] 训练数据来源、规模、混合比例写清楚
- [ ] 评测数字与 `eval/*.json` 一致（用 `tools/make_model_card.py` 生成，别手抄）
- [ ] 用法示例里的系统提示词与训练时**逐字一致**

## 3. 许可与合规

- [ ] `LICENSE` 是 Apache-2.0 全文（与底座一致）
- [ ] 保留了底座的 `NOTICE`/版权声明（若底座提供）
- [ ] 模型卡写明：老师模型只提供训练目标，权重未分发
- [ ] 训练数据里不含第三方版权文本（本项目种子语料为原创，老师数据由模型生成）

## 4. 上传

ModelScope（国内首选）：

```powershell
pip install modelscope
$env:MODELSCOPE_API_TOKEN = "在 https://www.modelscope.cn/my/myaccesstoken 生成"
python release/upload_modelscope.py --dir Qwen3.5-0.8B-TextPreprocessor `
    --repo Qwen3.5-0.8B-TextPreprocessor
```

Hugging Face：

```powershell
pip install huggingface_hub
$env:HF_TOKEN = "hf_xxx"
python release/upload_hf.py --dir Qwen3.5-0.8B-TextPreprocessor `
    --repo your-name/Qwen3.5-0.8B-TextPreprocessor
```

- [ ] 仓库名、license、pipeline_tag 与模型卡一致
- [ ] 上传后**在新环境**里按模型卡的示例跑一遍（最容易漏掉的是 chat template 没跟着走）
- [ ] 准备好回滚：保留一份本地 zip，上传出问题可以删库重传

## 5. 发布后

- [ ] 记录版本号与 commit（`merge_meta.json` 里有底座和适配器路径）
- [ ] 在 README 里给出复现命令，并注明实测硬件与耗时
- [ ] 收集前 20 条真实失败案例，写进"已知限制"，这比堆指标更能建立信任
