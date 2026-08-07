from __future__ import annotations

from typing import Sequence


class Qwen3Reranker:
    """Qwen3-Reranker adapter using the official yes/no likelihood formulation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "cuda",
        max_length: int = 8192,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install reranker-env dependencies") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map=device
        ).eval()
        self.max_length = max_length
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements based on '
            'the Query and the Instruct provided. Note that the answer can only be "yes" or '
            '"no".<|im_end|>\n<|im_start|>user\n'
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        prompts = [self._prompt(query, document) for document in documents]
        batch = self.tokenizer(
            prompts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for index, token_ids in enumerate(batch["input_ids"]):
            batch["input_ids"][index] = self.prefix_tokens + token_ids + self.suffix_tokens
        batch = self.tokenizer.pad(batch, padding=True, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            logits = self.model(**batch).logits[:, -1, [self.no_id, self.yes_id]]
            probabilities = self.torch.softmax(logits.float(), dim=-1)[:, 1]
        return probabilities.cpu().tolist()

    @staticmethod
    def _prompt(query: str, document: str) -> str:
        instruction = "Judge whether the scientific evidence answers the query."
        return (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}\n"
            "<Response>:"
        )
