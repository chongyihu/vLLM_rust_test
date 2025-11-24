from vllm import LLM, SamplingParams

# Load model once (fast for repeated calls)
llm = LLM("AMead10/Llama-3.2-3B-Instruct-AWQ")
params = SamplingParams(temperature=0.0, max_tokens=64)

def infer(prompt: str) -> dict:
    """Perform a single vLLM inference and return text, tokens_processed, and tokens_generated."""
    out = llm.generate([prompt], params)
    request_output = out[0]
    tokens_processed = len(request_output.prompt_token_ids)
    tokens_generated = len(request_output.outputs[0].token_ids)
    return {
        "text": request_output.outputs[0].text,
        "tokens_processed": tokens_processed,
        "tokens_generated": tokens_generated
    }
