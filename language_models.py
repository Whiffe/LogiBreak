# -*- coding: utf-8 -*-
"""LogiBreak 复现的模型封装。

  - HuggingFaceChat : 本地开源 chat 模型，优先用 modelscope 加载（缺失则回退
                      transformers）。每个实例只占用专属的 GPU，不与其它模型共享卡。

公开版仅保留本地模型后端，不读取或接收任何 API 密钥。论文原文部分配置使用托管模型，
因此本项目复现方法与流水线，不声称复现完全相同的模型配置。

生成时按 `max_batch` 把大批拆成小子批，避免一次前向把显存撑爆；子批遇到显存不足
(OOM) 会自动清缓存重试，再不行就对半拆分递归（等价于临时缩小 batch）。

注意：本文件刻意与 TAP 复现里的 language_models.py 保持一致，方便跨基线维护。
"""

import gc
import torch

try:  # 与既有代码库一致：优先用 modelscope 加载
    from modelscope import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - modelscope 不可用时回退 transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

# 显存不足(OOM)时每个子批最多重试的次数。
MAX_OOM_RETRIES = 3

# 不同 torch 版本里 OOM 异常类名不同，做个兼容。
try:
    _OOM_ERROR = torch.cuda.OutOfMemoryError
except AttributeError:  # pragma: no cover - 老版本回退到 RuntimeError
    _OOM_ERROR = RuntimeError


class HuggingFaceChat:
    """本地 causal LM，固定在专属 GPU 上，用 tokenizer 的 chat 模板。"""

    def __init__(self, model_path, dtype="auto", enable_thinking=False,
                 devices=0, max_batch=8, gpu_reserve_gb=2.0):
        """devices: 一个或多个「进程内」GPU 下标（即 CUDA_VISIBLE_DEVICES 重映射之后
        的 0..n-1）。单个下标 -> 整个模型钉在一张卡；列表 -> 只在这几张专属卡上铺开
        （适合思考模式、需要 >1 张卡的 8B 目标）。max_batch: 单次前向最多几条序列。"""
        self.model_path = model_path
        self.enable_thinking = enable_thinking
        self.max_batch = max(1, int(max_batch))

        dev_list = [devices] if isinstance(devices, int) else list(devices)

        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, "auto")

        if not torch.cuda.is_available():
            device_map, max_memory, where = "cpu", None, "cpu"
        elif len(dev_list) == 1:
            # 单卡：直接把整个模型放到这张卡，不用 device_map="auto"。
            device_map, max_memory = {"": int(dev_list[0])}, None
            where = f"cuda:{dev_list[0]}"
        else:
            # 多卡：在该模型专属的几张卡上铺开。由于每张卡只属于一个模型，可以按
            # 每卡总显存来分配预算（各预留约 gpu_reserve_gb GB）。
            device_map = "auto"
            reserve = int(gpu_reserve_gb * (1024 ** 3))
            max_memory = {
                int(d): max(0, torch.cuda.get_device_properties(int(d)).total_memory - reserve)
                for d in dev_list
            }
            where = "cuda:" + ",".join(str(d) for d in dev_list)
        print(f"[Load] Loading {model_path} -> {where} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
        ).eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # 输入 embedding 所在的卡；生成时输入张量放到这里。
        try:
            self.input_device = self.model.get_input_embeddings().weight.device
        except Exception:
            self.input_device = next(self.model.parameters()).device
        print(f"[Load] Done: {model_path}")

    def _apply_template(self, messages, assistant_prefix=None):
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **kwargs
            )
        except TypeError:
            # 不支持 enable_thinking 形参的老 tokenizer。
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        if assistant_prefix:
            text += assistant_prefix
        return text

    @torch.no_grad()
    def _generate_one_batch(self, texts, max_n_tokens, temperature, top_p):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}

        do_sample = temperature is not None and temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_n_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        outputs = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [o.strip() for o in outputs]

    @staticmethod
    def _free_cuda_cache():
        """清空 PyTorch 的显存缓存，给下一次重试腾出空间。"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_chunk_with_retry(self, chunk, max_n_tokens, temperature, top_p):
        """生成一个子批；遇到显存不足(OOM)时清空显存缓存后最多重试
        MAX_OOM_RETRIES 次。若重试仍失败且子批多于一条，则把子批对半拆分递归
        （等价于临时调小 batch），直到拆到单条仍 OOM 才向上抛出异常。"""
        last_err = None
        for attempt in range(1, MAX_OOM_RETRIES + 1):
            try:
                return self._generate_one_batch(
                    chunk, max_n_tokens, temperature, top_p)
            except _OOM_ERROR as e:
                last_err = e
                self._free_cuda_cache()
                print(f"[OOM] 子批大小={len(chunk)}，第 {attempt}/{MAX_OOM_RETRIES} "
                      f"次重试（已清空显存缓存）...")
        # 重试用尽：能拆就对半拆分递归，否则只能放弃。
        if len(chunk) > 1:
            mid = len(chunk) // 2
            print(f"[OOM] 重试 {MAX_OOM_RETRIES} 次仍失败，将子批 {len(chunk)} "
                  f"拆为 {mid}+{len(chunk) - mid} 继续 ...")
            left = self._generate_chunk_with_retry(
                chunk[:mid], max_n_tokens, temperature, top_p)
            right = self._generate_chunk_with_retry(
                chunk[mid:], max_n_tokens, temperature, top_p)
            return left + right
        raise last_err

    def batched_generate(self, messages_list, max_n_tokens, temperature, top_p,
                         assistant_prefix=None):
        """渲染 + 生成，按 self.max_batch 拆成小子批。

        每个子批带 OOM 自动重试：显存不足时清缓存重试 3 次，仍失败则对半拆分。"""
        texts = [self._apply_template(m, assistant_prefix) for m in messages_list]
        results = []
        for i in range(0, len(texts), self.max_batch):
            chunk = texts[i:i + self.max_batch]
            results.extend(
                self._generate_chunk_with_retry(
                    chunk, max_n_tokens, temperature, top_p)
            )
        return results
