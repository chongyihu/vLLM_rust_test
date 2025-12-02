from vllm import LLM, SamplingParams
# Load model once (fast for repeated calls)
llm = LLM("microsoft/phi-4", tensor_parallel_size=2,enable_prefix_caching=True)
params = SamplingParams(temperature=0.8, max_tokens=2000)

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

def cleanup():
    """Clean up the vLLM engine and release GPU memory."""
    global llm
    if llm is not None:
        try:
            # Shutdown the LLM engine to release GPU memory
            if hasattr(llm, 'llm_engine') and llm.llm_engine is not None:
                llm.shutdown()
            # Also try to delete the model to free memory
            del llm
            llm = None
            # Force garbage collection
            import gc
            gc.collect()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            llm = None
