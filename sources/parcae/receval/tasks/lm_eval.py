import torch
from typing import Optional, Any
from dataclasses import dataclass
from tqdm import tqdm

from lm_eval.api.model import LM

from receval.settings import CLISettings, LMEvalTaskSettings


@dataclass
class LMEvalResults:
    results: dict[str, Any]
    task_scores: dict[str, float]
    aggregate_score: Optional[float] = None


class LMEvalModel(LM):
    def __init__(self, model, tokenizer, settings: CLISettings):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.device = settings.device
        self.batch_size = settings.tasks.lm_eval.batch_size
        self._rank = settings.ddp_rank
        self._world_size = settings.ddp_world_size
        from accelerate import Accelerator
        self.accelerator = Accelerator()

    @property
    def eot_token_id(self):
        if hasattr(self.tokenizer, 'eos_token_id'):
            return self.tokenizer.eos_token_id
        return self.tokenizer.eos_id

    @property
    def max_length(self):
        return self.settings.sequence_length or 2048

    @property
    def max_gen_toks(self):
        return 256

    def tok_encode(self, string: str) -> list[int]:
        if hasattr(self.tokenizer, 'eos_id'):
            result = self.tokenizer.encode(string, return_tensors=False)
        else:
            result = self.tokenizer.encode(string, add_special_tokens=False)
        return result if isinstance(result, list) else result.tolist()

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with self.settings._get_autocast_context():
                outputs = self.model.forward_for_generation(input_ids)
                return outputs["logits"]

    def _model_call_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Use the original forward pass (same as training/val_loss)."""
        with torch.no_grad():
            with self.settings._get_autocast_context():
                outputs = self.model.forward(input_ids, return_logits=True)
                return outputs["logits"]

    def _model_generate(self, input_ids: torch.Tensor, max_tokens: int, stop_tokens: list[int]) -> torch.Tensor:
        with torch.no_grad():
            with self.settings._get_autocast_context():
                return self.model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                    do_sample=False,
                )

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        results = []
        for req in tqdm(requests, desc="loglikelihood", disable=self._rank != 0):
            context, continuation = req.args
            ctx_tokens = self.tok_encode(context)
            full_tokens = self.tok_encode(context + continuation)
            if ctx_tokens != full_tokens[:len(ctx_tokens)]:
                context = context.rstrip()
                ctx_tokens = self.tok_encode(context)
                full_tokens = self.tok_encode(context + continuation)
            cont_tokens = full_tokens[len(ctx_tokens):]
            if len(cont_tokens) == 0:
                results.append((0.0, True))
                continue
            if len(full_tokens) > self.max_length:
                overflow = len(full_tokens) - self.max_length
                ctx_tokens = ctx_tokens[overflow:]
                full_tokens = full_tokens[overflow:]
            input_ids = torch.tensor([full_tokens], device=self.device)
            logits = self._model_call(input_ids)
            log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
            cont_start = len(ctx_tokens)
            cont_log_prob = 0.0
            for i, token in enumerate(cont_tokens):
                pos = cont_start + i - 1
                if pos >= 0:
                    cont_log_prob += log_probs[pos, token].item()
            greedy_tokens = logits[0, max(0, cont_start - 1):len(full_tokens) - 1].argmax(dim=-1).tolist()
            is_greedy = greedy_tokens == cont_tokens
            results.append((cont_log_prob, is_greedy))
        return results

    def loglikelihood_rolling(self, requests) -> list[float]:
        results = []
        for req in tqdm(requests, desc="loglikelihood_rolling", disable=self._rank != 0):
            (string,) = req.args
            tokens = self.tok_encode(string)
            total_log_prob = 0.0
            max_len = self.max_length
            prev_last_log_probs = None  # Store last log_probs row from previous chunk
            
            for start in range(0, len(tokens), max_len):
                chunk = tokens[start:start + max_len]
                if len(chunk) == 0:
                    continue
                    
                input_ids = torch.tensor([chunk], device=self.device)
                # Use original forward pass (same as training/val_loss)
                logits = self._model_call_forward(input_ids)
                log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
                
                # For first chunk: skip token 0 (no prior context)
                # For subsequent chunks: use prev_last_log_probs to score token 0
                if start == 0:
                    # Skip first token - we have no prior context to predict it
                    for i in range(1, len(chunk)):
                        total_log_prob += log_probs[i - 1, chunk[i]].item()
                else:
                    # Use last log_probs from previous chunk to score first token
                    if prev_last_log_probs is not None and len(chunk) > 0:
                        total_log_prob += prev_last_log_probs[chunk[0]].item()
                    # Score remaining tokens in this chunk
                    for i in range(1, len(chunk)):
                        total_log_prob += log_probs[i - 1, chunk[i]].item()
                
                # Store last row of log_probs for next chunk
                prev_last_log_probs = log_probs[-1]
                
            results.append(total_log_prob)
        return results

    def generate_until(self, requests) -> list[str]:
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            ctx_tokens = self.tok_encode(context)
            input_ids = torch.tensor([ctx_tokens], device=self.device)
            max_tokens = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            stop_tokens = gen_kwargs.get("until", [self.eot_token_id])
            if stop_tokens and isinstance(stop_tokens[0], str):
                stop_tokens = [self.tok_encode(s)[0] for s in stop_tokens if s]
            output = self._model_generate(input_ids, max_tokens, stop_tokens)
            generated = output[0, len(ctx_tokens):].tolist()
            text = self.tok_decode(generated)
            for stop in gen_kwargs.get("until", []):
                if stop in text:
                    text = text[:text.index(stop)]
            results.append(text)
        return results


def run_lm_eval(model, tokenizer, settings: CLISettings) -> LMEvalResults:
    try:
        import lm_eval
        from lm_eval import evaluator
    except ImportError:
        raise ImportError("lm-evaluation-harness not installed. Run: pip install lm-eval")

    task_settings = settings.tasks.lm_eval
    lm = LMEvalModel(model, tokenizer, settings)

    results = evaluator.simple_evaluate(
        model=lm,  # type: ignore
        tasks=task_settings.tasks,  # type: ignore
        num_fewshot=task_settings.num_fewshot,
        batch_size=task_settings.batch_size,
        limit=task_settings.limit,
    )

    raw_results = results if results else {}
    task_scores = raw_results.get("results", {})

    return LMEvalResults(
        results=raw_results,
        task_scores=task_scores,
        aggregate_score=None,
    )


def run_lm_eval_simple(
    model,
    tokenizer,
    tasks: list[str],
    num_fewshot: int = 0,
    batch_size: int = 32,
    limit: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> LMEvalResults:
    try:
        from lm_eval import evaluator
    except ImportError:
        raise ImportError("lm-evaluation-harness not installed. Run: pip install lm-eval")

    if device is None:
        device = next(model.parameters()).device

    class SimpleLMWrapper:
        def __init__(self, model, tokenizer, device, batch_size):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.batch_size = batch_size

        @property
        def eot_token_id(self):
            if hasattr(self.tokenizer, 'eos_token_id'):
                return self.tokenizer.eos_token_id
            return self.tokenizer.eos_id

        @property
        def max_length(self):
            return 2048

        @property
        def max_gen_toks(self):
            return 256

        def tok_encode(self, string: str) -> list[int]:
            result = self.tokenizer.encode(string, return_tensors=False)
            return result if isinstance(result, list) else result.tolist()

        def tok_decode(self, tokens: list[int]) -> str:
            return self.tokenizer.decode(tokens)

        def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                outputs = self.model.forward_for_generation(input_ids)
                return outputs["logits"]

        def _model_call_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            """Use the original forward pass (same as training/val_loss)."""
            with torch.no_grad():
                outputs = self.model.forward(input_ids, return_logits=True)
                return outputs["logits"]

        def loglikelihood(self, requests) -> list[tuple[float, bool]]:
            results = []
            for context, continuation in requests:
                ctx_tokens = self.tok_encode(context)
                cont_tokens = self.tok_encode(continuation)
                full_tokens = ctx_tokens + cont_tokens
                input_ids = torch.tensor([full_tokens], device=self.device)
                logits = self._model_call(input_ids)
                log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
                cont_start = len(ctx_tokens)
                cont_log_prob = 0.0
                for i, token in enumerate(cont_tokens):
                    cont_log_prob += log_probs[cont_start + i - 1, token].item()
                greedy_tokens = logits[0, cont_start - 1:-1].argmax(dim=-1).tolist()
                is_greedy = greedy_tokens == cont_tokens
                results.append((cont_log_prob, is_greedy))
            return results

        def loglikelihood_rolling(self, requests) -> list[float]:
            results = []
            for (string,) in requests:
                tokens = self.tok_encode(string)
                total_log_prob = 0.0
                max_len = self.max_length
                prev_last_log_probs = None
                
                for start in range(0, len(tokens), max_len):
                    chunk = tokens[start:start + max_len]
                    if len(chunk) == 0:
                        continue
                        
                    input_ids = torch.tensor([chunk], device=self.device)
                    # Use original forward pass (same as training/val_loss)
                    logits = self._model_call_forward(input_ids)
                    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
                    
                    if start == 0:
                        for i in range(1, len(chunk)):
                            total_log_prob += log_probs[i - 1, chunk[i]].item()
                    else:
                        if prev_last_log_probs is not None and len(chunk) > 0:
                            total_log_prob += prev_last_log_probs[chunk[0]].item()
                        for i in range(1, len(chunk)):
                            total_log_prob += log_probs[i - 1, chunk[i]].item()
                    
                    prev_last_log_probs = log_probs[-1]
                    
                results.append(total_log_prob)
            return results

        def generate_until(self, requests) -> list[str]:
            results = []
            for context, gen_kwargs in requests:
                ctx_tokens = self.tok_encode(context)
                input_ids = torch.tensor([ctx_tokens], device=self.device)
                max_tokens = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
                output = self.model.generate(input_ids, max_new_tokens=max_tokens, temperature=0.0, do_sample=False)
                generated = output[0, len(ctx_tokens):].tolist()
                text = self.tok_decode(generated)
                for stop in gen_kwargs.get("until", []):
                    if stop in text:
                        text = text[:text.index(stop)]
                results.append(text)
            return results

    lm = SimpleLMWrapper(model, tokenizer, device, batch_size)
    results = evaluator.simple_evaluate(model=lm, tasks=tasks, num_fewshot=num_fewshot, batch_size=batch_size, limit=limit)  # type: ignore

    raw_results = results if results else {}
    task_scores = raw_results.get("results", {})
    return LMEvalResults(results=raw_results, task_scores=task_scores, aggregate_score=None)
