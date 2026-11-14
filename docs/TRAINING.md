# 训练环境与踩坑记录

本项目的训练是在一台 **Windows 11 + RTX 4060 8GB** 的消费级机器上完成的。
记录几个会直接影响成败的点，方便后来者少走弯路。

## 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop，8GB，驱动 CUDA 13.0 |
| Python | 3.10.22 |
| torch | 2.11.0+cu128 |
| transformers | 5.20.2 |
| peft | 0.21.3 |
| triton | 3.8.0（`triton-windows`） |
| flash-linear-attention | 0.6.0 |

## 坑 1：Qwen3.5 用的是混合线性注意力，没有内核会慢 20 倍

`Qwen3_5ForConditionalGeneration` 的 24 层里有 18 层是 `linear_attention`（gated delta rule）+
`causal_conv`。torch 里没有原生算子，transformers 会回退到参考实现并打印：

```
`chunk_gated_delta_rule` is falling back to its reference PyTorch implementation ...
```

实测单步 **21.7s**。装上 `triton-windows` + `flash-linear-attention` 后降到 11.5s，
再叠加 batch 调整后到 4.1s/步（batch=1、梯度累积 16、序列 512）。

Windows 上没有官方 `causal_conv1d` 轮子（需要本地编译 CUDA 扩展），
这一项仍然是参考实现，但相比 delta rule 影响小得多。

## 坑 2：8GB 显存 + Windows WDDM = 训练"假死"

batch=2 时 PyTorch 预留显存会爬到 **7.3-7.8 GiB**，超过可用显存后 Windows 不会 OOM，
而是退到**共享内存**（系统 RAM 当显存用）。表现是：

- `nvidia-smi` 显示 GPU 利用率 100%、显存占满；
- 但 step 计数器**完全不动**，几十分钟没有任何 checkpoint；
- 进程 CPU 时间很低，日志不再增长。

这不是死锁也不是代码 bug，是显存溢出的病态状态。两个对策：

1. `torch.cuda.set_per_process_memory_fraction(0.92)`——把"悄悄溢出"变成"显式 OOM"，
   问题立刻暴露而不是耗掉一晚上（本项目的 `--mem-fraction` 参数）；
2. 降 batch。实测这个模型在 8GB 上 **batch=1 反而比 batch=2 快**（0.26 vs 0.29 秒/样本），
   因为 batch=2 触发了显存回收与分页。最终配置就是 batch=1 + 梯度累积 16。

另外 `expandable_segments:True` 在 Windows 上**不支持**，别指望它治碎片：

```
UserWarning: expandable_segments not supported on this platform
```

## 坑 3：显存不是瓶颈，序列长度才是

训练集 3000 条样本的 token 长度：均值 186，p50 162，p95 405，最大 913。
所以 `max_len=512` 只会丢掉约 2-3% 的长样本，却能让显存和步时都降下来。
把 `max_len` 设成 1024 只是在给少数长样本买单。

## 坑 4：聊天模板会自动包一个空的思考块

Qwen3.5 的 `chat_template.jinja` 在 assistant 段前会渲染：

```
<|im_start|>assistant
<think>

</think>

```

也就是说模型被训练成"先输出一个空的思考块，再输出正文"。训练时只要用
`apply_chat_template(messages[:-1], add_generation_prompt=True)` 作为前缀去 mask loss，
这个空块就会被正确跳过；推理时同样用 `add_generation_prompt=True`，模型直接续写正文，
不会吐出 `<think>` 标签。**不需要**手动在 prompt 里拼标签。

## 坑 5：别教 0.8B 模型修乱码——它会编造整段内容

第一版把"UTF-8 被按 Latin-1 解码"的乱码修复也放进了任务里，训练数据中约 8% 是
"乱码 → 原文"的样本。真实测试时出现：

```
输入: 小区物业通知：本周四上午九点到下午五点进行二次供水水箱清洗……
输出: 车间安全检查要点：本周六下午三点到下月五点执行三次安全大检查……
```

模型不是"修错了"，而是**另写了一段看起来像中文的内容**。原因很清楚：字节级还原需要
逐字符重排，0.8B 模型学不会，于是走了捷径——记住"看到乱码就输出某段训练集里的中文"。
种子语料只有 130 篇，这个捷径在训练集上 loss 极低，一上真实输入就暴露。

结论：**字节层的事交给确定性代码**。`repair_mojibake()` 用
`encode('latin-1').decode('utf-8')` 做还原，几微秒、零幻觉。模型训练数据里彻底移除了乱码。

## 坑 6：数据里没有"什么都不用做"的样本，模型就会乱动

第一版为了"每个样本都有信息量"，把 `clean(raw) == raw` 的样本全丢弃了。结果模型学到的是
"凡输入皆要动手"：

- 把合法的两字叠词压掉：`刚刚` → `刚`
- 给短句凭空补字：`需要我做什么` → `需要我做点什么`

对清洗模型来说，**"不动"和"动手"一样是需要学习的动作**。修复办法：

1. 22% 的样本改成 identity（脏文本 = 干净文本），并专门加入"刚刚/谢谢/妈妈/看看"这类
   含合法叠词的种子；
2. 验证集里保留 identity 样本，否则模型会偏向"多改"。

## 坑 7：标签污染不会在 loss 上报警

乱码样本被污染时（规则把 CP1252 乱码"修"成了 GB18030 生僻字），模型学到的是
"输出一串谁都看不懂的字"。这种问题 loss 照样降得很好看，必须靠数据审计发现——
按字符集和常用字比例抽查标签，比看 loss 曲线有用得多。

## 最终训练配置

| 项 | 值 |
|---|---|
| 数据 | 11622 条（规则 9704 + 老师 1918），验证 200 条 |
| batch × 累积 | 1 × 16 |
| max_len | 512 |
| LoRA | r=16, alpha=32, dropout=0.05, 7.27M 可训练参数（0.85%） |
| 学习率 | 1e-4，cosine，warmup 3% |
| epoch | 2（1426 步） |
| 步时 | ≈4.1 s |
| 显存 | 分配 1.69 GiB / 预留 3.50 GiB |
| 总时长 | ≈1.6 小时 |
