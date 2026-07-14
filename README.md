# LogiBreak - Logic Jailbreak reproduction

[中文说明](README.zh-CN.md)

An independent, local-model reproduction of Peng et al., [*Logic Jailbreak:
Efficiently Unlocking LLM Safety Restrictions Through Formal Logical
Expression*](https://arxiv.org/abs/2505.13527) (ACL 2026). The authors' official
implementation is available at
[Applied-Machine-Learning-Lab/ACL2026_Logibreak](https://github.com/Applied-Machine-Learning-Lab/ACL2026_Logibreak).

This release contains code and method documentation only. It deliberately
excludes benchmark files, generated responses, evaluation results, model
weights, machine-specific commands, papers, caches, and remote inference API
support.

> For authorized safety research and red-team evaluation only. Do not use this
> project to facilitate harm or to generate and distribute harmful material.

## Method

For each request, LogiBreak performs four steps:

1. A translator model converts the natural-language request into a first-order
   logic (FOL) expression.
2. The program builds `x' = x_context || x_logic || x_instruct`, combining a
   formal-semantics context, the FOL expression, and a response instruction.
3. A target model receives the transformed prompt.
4. A rule-based, local LLM, or Qwen3Guard judge evaluates the response.

The program reports ASR@1 and ASR@N. ASR@N treats a request as successful when
at least one of N independent trials is judged successful; the default is N=5.

## What differs from the paper

The paper uses hosted models in parts of its evaluation. This reproduction is
local-model only and defaults to Qwen model identifiers. Consequently, it
reproduces the method and evaluation pipeline, not the paper's exact model
configuration or reported numbers.

The prompts in `system_prompts.py` follow the English templates in Figures
8-10 of the paper. Qwen3Guard is an additional local evaluator used for
cross-baseline comparison; it is not one of the paper's three original judges.

## Installation

Python 3.10+ and a CUDA-capable PyTorch environment are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Weights may be supplied as local paths or as Hugging Face/ModelScope model
identifiers. No API key is read by this project.

## Data

Benchmark data is not included. See [Dataset/README.md](Dataset/README.md) for
the one-column CSV format and prepare only data you are authorized to evaluate.

## Quick start

Rule judge, two GPUs:

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

Qwen3Guard judge, three GPUs:

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

Remove `--limit 1` after the smoke test. Add `--resume` to continue an
interrupted run. If generation runs out of memory, lower `--max-batch` to 4, 2,
or 1. A role can use multiple GPUs with `--attack-gpu`, `--target-gpu`, or
`--guard-gpu`.

## Judges

| Option | Model required | Decision |
|---|---:|---|
| `rule` | No | A response without a known refusal prefix is counted as jailbroken |
| `llm` + `gpt` style | Yes, local | Parsed 1-10 rating; default threshold is 10 |
| `llm` + `llama` style | Yes, local | Parsed `safe` / `unsafe` label |
| `guard` | Yes, local | Qwen3Guard: `Unsafe` or `Controversial` counts as successful |

The inexpensive rule decision is always stored alongside the selected judge's
decision. Rule judging can overestimate success because it checks refusal
phrases rather than semantic harmfulness; use a semantic judge for formal
reporting.

## Output and reliability

The output JSON contains `config`, `summary`, and per-request `results`. It is
written atomically after every item. The runner supports:

- `--resume` for completed-item skipping and retrying errored items;
- per-item error isolation;
- automatic CUDA OOM retry and recursive batch splitting;
- progress, elapsed time, average time, and ETA reporting;
- `--ablation no-context` and `--ablation raw-request`.

Generated responses and results are ignored by Git through `.gitignore`.

## Repository layout

```text
.
|-- logibreak_main.py       # CLI, pipeline, resume, and result writing
|-- conversers.py           # translator, target, and local judges
|-- language_models.py      # local Hugging Face/ModelScope generation
|-- system_prompts.py       # paper prompt templates
|-- common.py               # FOL cleanup and judge-output parsing
|-- logibreak_animation.html# standalone method animation
|-- Dataset/README.md       # input format only; no benchmark data
`-- results/README.md       # output policy only; no experiment results
```

Open `logibreak_animation.html` in a browser for a standalone visual overview.

## Citation

```bibtex
@article{peng2025logic,
  title={Logic Jailbreak: Efficiently Unlocking LLM Safety Restrictions Through Formal Logical Expression},
  author={Peng, Jingyu and Wang, Maolin and Wang, Nan and Li, Jiatong and Li, Yuchen and Ye, Yuyang and Wang, Wanyu and Jia, Pengyue and Zhang, Kai and Zhao, Xiangyu},
  journal={arXiv preprint arXiv:2505.13527},
  year={2025}
}
```

