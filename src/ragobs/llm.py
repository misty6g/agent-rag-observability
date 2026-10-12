from __future__ import annotations
import time
from dataclasses import dataclass

@dataclass
class Generation:
    text: str
    tokens_in: int
    tokens_out: int
    seconds: float

class HFChatLLM:
    """Local instruct model via HF transformers on CPU, greedy decoding."""
    def __init__(self, name: str = "Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens: int = 48, dtype: str = "float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(0)
        torch.set_num_threads(4)
        self.torch = torch
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype)).eval()
        self.max_new_tokens = max_new_tokens
    def generate(self, system: str, user: str) -> Generation:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt")
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        dt = time.perf_counter() - t0
        new = out[0, ids["input_ids"].shape[1]:]
        return Generation(self.tok.decode(new, skip_special_tokens=True).strip(),
                          int(ids["input_ids"].shape[1]), int(new.shape[0]), dt)

class EchoLLM:
    """Deterministic stand-in for tests: returns the first context line."""
    name = "echo"
    def generate(self, system: str, user: str) -> Generation:
        ctx = user.split("Context:", 1)[-1].strip()
        txt = ctx.split("\n", 1)[0][:200]
        return Generation(txt, len(user.split()), len(txt.split()), 0.001)
