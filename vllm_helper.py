from vllm import LLM, SamplingParams

# Load model once (fast for repeated calls)
llm = LLM("AMead10/Llama-3.2-3B-Instruct-AWQ")
params = SamplingParams(temperature=0.0, max_tokens=64)

def infer(prompt: str) -> str:
    """Perform a single vLLM inference."""
    out = llm.generate([prompt], params)
    return out[0].outputs[0].text
