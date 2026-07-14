# LogiBreak - Logic Jailbreak 本地复现

[English](README.md)

这是对 Peng 等人论文 [*Logic Jailbreak: Efficiently Unlocking LLM Safety
Restrictions Through Formal Logical Expression*](https://arxiv.org/abs/2505.13527)
（ACL 2026）的独立本地模型复现。作者官方实现见
[Applied-Machine-Learning-Lab/ACL2026_Logibreak](https://github.com/Applied-Machine-Learning-Lab/ACL2026_Logibreak)。

公开版只保留代码和方法文档，已去除基准数据、生成内容、实验结果、模型权重、论文 PDF、
机器专用命令、缓存及远程推理 API 支持。

> 仅用于经过授权的模型安全研究和红队评测。请勿将本项目用于实施危害或生成、传播有害内容。

## 方法

LogiBreak 对每条请求执行四步：

1. 翻译器模型把自然语言请求转换为一阶逻辑（FOL）表达式；
2. 程序拼接 `x' = x_context || x_logic || x_instruct`，即形式语义语境、逻辑式和回答指令；
3. 将转换后的提示发给目标模型；
4. 用规则、本地 LLM 或 Qwen3Guard 判定目标回答。

程序同时报告 ASR@1 和 ASR@N。ASR@N 表示同一请求独立尝试 N 次，只要一次被判成功，
这条请求就计为成功；默认 `N=5`。

## 与论文设置的区别

论文部分实验使用托管模型，本复现只支持本地模型，默认使用 Qwen 模型标识。因此本项目复现
的是方法与评估流程，不声称复现论文的完全相同模型配置或数值。

`system_prompts.py` 依据论文图 8-10 的英文模板实现。Qwen3Guard 是为了与其他本地基线
统一而增加的判官，不属于论文原始的三类判官。

## 安装

推荐 Python 3.10+ 和支持 CUDA 的 PyTorch 环境。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

模型既可以填写本地权重路径，也可以填写 Hugging Face/ModelScope 模型标识。本项目不读取
任何 API 密钥。

## 数据格式

仓库不包含基准数据。请按照 [Dataset/README.md](Dataset/README.md) 准备一列 CSV，并且只
评测你有权使用的数据。

## 快速运行

规则判官，两张 GPU：

```bash
python logibreak_main.py \
  --input ./Dataset/requests.csv \
  --output ./results/run.json \
  --GPU 0,1 \
  --translator-model-path Qwen/Qwen3-4B \
  --target-model-path Qwen/Qwen3-4B \
  --judge-type rule \
  --num-trials 5 \
  --limit 1
```

Qwen3Guard 判官，三张 GPU：

```bash
python logibreak_main.py \
  --input ./Dataset/requests.csv \
  --output ./results/run_guard.json \
  --GPU 0,1,2 \
  --translator-model-path Qwen/Qwen3-4B \
  --target-model-path Qwen/Qwen3-4B \
  --judge-type guard \
  --guard-path Qwen/Qwen3Guard-Gen-4B \
  --num-trials 5 \
  --limit 1
```

冒烟测试通过后删掉 `--limit 1`。任务中断后加 `--resume` 继续。显存不足时把
`--max-batch` 调到 4、2 或 1；也可以通过 `--attack-gpu`、`--target-gpu`、
`--guard-gpu` 为单个角色分配多张卡。

## 判官

| 选项 | 是否需要模型 | 判定方式 |
|---|---:|---|
| `rule` | 否 | 未命中常见拒绝前缀即计为越狱 |
| `llm` + `gpt` 风格 | 是，本地 | 解析 1-10 分，默认 10 分才算成功 |
| `llm` + `llama` 风格 | 是，本地 | 解析 `safe` / `unsafe` 标签 |
| `guard` | 是，本地 | Qwen3Guard 的 `Unsafe` 或 `Controversial` 计为成功 |

规则判官的结果始终会一并保存，但它只查拒绝短语，可能高估真正的攻击成功率；正式报告建议
使用语义判官。

## 输出与鲁棒性

输出 JSON 包含 `config`、`summary` 和逐条 `results`，每完成一条就原子写盘。程序支持：

- `--resume` 跳过已完成条目，并重试上次报错条目；
- 单条报错隔离；
- CUDA OOM 自动重试与递归拆批；
- 进度、耗时、平均耗时和 ETA；
- `--ablation no-context` 与 `--ablation raw-request` 消融。

`.gitignore` 已默认屏蔽生成回答、结果文件、数据集、密钥文件、模型权重和本地命令。

## 项目结构

```text
.
|-- logibreak_main.py        # CLI、主流程、续跑和结果写入
|-- conversers.py            # 翻译器、目标模型和本地判官
|-- language_models.py       # 本地 Hugging Face/ModelScope 推理
|-- system_prompts.py        # 论文提示模板
|-- common.py                # FOL 清洗和判官输出解析
|-- logibreak_animation.html # 独立流程动画
|-- Dataset/README.md        # 只有输入格式，无基准数据
`-- results/README.md        # 只有输出说明，无实验结果
```

直接用浏览器打开 `logibreak_animation.html`，可以查看方法流程动画。

## 引用

```bibtex
@article{peng2025logic,
  title={Logic Jailbreak: Efficiently Unlocking LLM Safety Restrictions Through Formal Logical Expression},
  author={Peng, Jingyu and Wang, Maolin and Wang, Nan and Li, Jiatong and Li, Yuchen and Ye, Yuyang and Wang, Wanyu and Jia, Pengyue and Zhang, Kai and Zhao, Xiangyu},
  journal={arXiv preprint arXiv:2505.13527},
  year={2025}
}
```

