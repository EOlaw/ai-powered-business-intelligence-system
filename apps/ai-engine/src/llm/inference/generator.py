"""
InsightSerenity AI Engine — Text Generator
===========================================
The inference engine that orchestrates the complete text generation pipeline:
    prompt → tokenizer → model (with KV cache) → sampler → tokenizer → text

This is what you call to get text out of our LLM at inference time.
No external API. The model runs locally on our own weights.

Generation modes:
    1. generate(prompt) → str
       Generates the full response at once, returns it as a string.

    2. stream_generate(prompt) → Iterator[str]
       Yields one token at a time as they are generated.
       Used for streaming responses to clients (SSE or WebSocket).

Autoregressive decoding loop:
    1. Encode the prompt → input_ids
    2. Forward pass (prefill): process all prompt tokens, cache K,V
    3. Get logits at the last position → sample next token
    4. Append new token to input_ids
    5. Forward pass (decode): single token, use KV cache
    6. Repeat from step 3 until EOS or max_new_tokens

Stopping conditions:
    - EOS token generated
    - max_new_tokens reached
    - Stop strings matched in generated text

Usage:
    from src.llm.inference.generator import TextGenerator

    generator = TextGenerator(model=gpt, tokenizer=bpe_tokenizer, device="cuda")

    # Single response
    text = generator.generate("What is gravity?", max_new_tokens=200)

    # Streaming
    for token in generator.stream_generate("Explain quantum physics"):
        print(token, end="", flush=True)
"""

from typing import Any, Iterator, List, Optional

import torch
from torch import Tensor

from src.llm.inference.sampler import Sampler
from src.tokenizer.special_tokens import SpecialTokens as ST
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextGenerator:
    """
    Autoregressive text generator using our own LLM.

    Args:
        model:          The GPTDecoder model (must be already loaded and eval'd).
        tokenizer:      Tokenizer with encode/decode methods.
        device:         Compute device for inference. Default "cpu".
        max_seq_len:    Maximum total sequence length (prompt + response).
    """

    def __init__(
        self,
        model:       Any,
        tokenizer:   Any,
        device:      str = "cpu",
        max_seq_len: int = 2048,
    ) -> None:
        self.model       = model
        self.tokenizer   = tokenizer
        self.device      = torch.device(device)
        self.max_seq_len = max_seq_len

        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt:         str,
        max_new_tokens: int   = 256,
        strategy:       str   = "top_p",
        temperature:    float = 1.0,
        top_k:          int   = 50,
        top_p:          float = 0.9,
        rep_penalty:    float = 1.1,
        stop_strings:   Optional[List[str]] = None,
    ) -> str:
        """
        Generate text from a prompt and return the complete response.

        Args:
            prompt:         Input text prompt.
            max_new_tokens: Maximum tokens to generate after the prompt.
            strategy:       "greedy", "top_k", or "top_p".
            temperature:    Logit temperature scaling.
            top_k:          K for top-k sampling.
            top_p:          p for nucleus sampling.
            rep_penalty:    Repetition penalty (1.0 = off).
            stop_strings:   Optional list of strings — stop generation when any appears.

        Returns:
            Generated text string (prompt not included).
        """
        tokens = list(self.stream_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rep_penalty=rep_penalty,
            stop_strings=stop_strings,
        ))
        return "".join(tokens)

    def stream_generate(
        self,
        prompt:         str,
        max_new_tokens: int   = 256,
        strategy:       str   = "top_p",
        temperature:    float = 1.0,
        top_k:          int   = 50,
        top_p:          float = 0.9,
        rep_penalty:    float = 1.1,
        stop_strings:   Optional[List[str]] = None,
    ) -> Iterator[str]:
        """
        Generate text token-by-token, yielding each decoded token string.

        This is the streaming interface used by the inference API to send
        server-sent events to clients as tokens are generated.

        Args:
            Same as generate().

        Yields:
            Decoded token strings, one at a time.
        """
        sampler = Sampler(
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rep_penalty=rep_penalty,
        )

        # Encode prompt (no BOS — the formatter already adds role tokens)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            return

        # Initialise input tensor
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self.device
        )   # (1, T_prompt)

        generated_ids: List[int] = []
        generated_text = ""

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass: full sequence on first step, then single token
                output = self.model(input_ids)
                logits = output["logits"]   # (1, T, V)

                # Sample from the logits at the LAST position
                next_logits = logits[0, -1, :]   # (V,)

                # Apply repetition penalty using all tokens so far
                all_ids_so_far = torch.tensor(
                    prompt_ids + generated_ids,
                    dtype=torch.long,
                    device=self.device,
                )

                next_token_id = sampler.sample(next_logits, all_ids_so_far)

                # Stop conditions
                if next_token_id == ST.EOS_ID:
                    break
                if next_token_id == ST.END_TURN_ID:
                    break

                generated_ids.append(next_token_id)

                # Decode the new token
                new_token_str = self.tokenizer.decode(
                    [next_token_id], skip_special_tokens=True
                )
                generated_text += new_token_str

                yield new_token_str

                # Check stop strings
                if stop_strings:
                    if any(s in generated_text for s in stop_strings):
                        break

                # Append new token for the next step
                # Note: in a production system you'd use a KV cache here
                # to avoid re-processing the entire sequence every step.
                # The KVCache class is ready in kv_cache.py and the model
                # would need to be modified to accept and update it.
                next_token_tensor = torch.tensor(
                    [[next_token_id]], dtype=torch.long, device=self.device
                )
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

                # Safety: don't exceed model's max sequence length
                if input_ids.shape[1] >= self.max_seq_len:
                    logger.debug(
                        "Max sequence length reached, stopping generation",
                        step=step,
                    )
                    break

    def generate_batch(
        self,
        prompts:        List[str],
        max_new_tokens: int   = 256,
        strategy:       str   = "top_p",
        temperature:    float = 1.0,
        top_k:          int   = 50,
        top_p:          float = 0.9,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts independently.

        Note: true batch generation (processing multiple prompts in parallel)
        requires padding all prompts to the same length. This implementation
        generates them sequentially — batch parallelism is handled by the
        serving layer via dynamic batching.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Token budget per prompt.
            Other args: same as generate().

        Returns:
            List of generated strings, one per prompt.
        """
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            results.append(result)
        return results
