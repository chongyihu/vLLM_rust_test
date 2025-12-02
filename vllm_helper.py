import os
import sys
import multiprocessing

# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch or vLLM
# Once CUDA is initialized, changing CUDA_VISIBLE_DEVICES has no effect
current_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')
print(f"Current CUDA_VISIBLE_DEVICES: {current_cuda_visible}")

# If not set, configure it for 2 GPUs (5090x2)
if current_cuda_visible == 'NOT SET' or current_cuda_visible == '':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    print(f"Setting CUDA_VISIBLE_DEVICES=0,1 for tensor parallelism")
else:
    print(f"Using existing CUDA_VISIBLE_DEVICES={current_cuda_visible}")
    gpu_list = [x.strip() for x in current_cuda_visible.split(',') if x.strip()]
    if len(gpu_list) < 2:
        print(f"WARNING: CUDA_VISIBLE_DEVICES only specifies {len(gpu_list)} GPU(s), but tensor_parallel_size=2")
        print("This may cause the CUDA error you're seeing.")
        print("Solution: Set CUDA_VISIBLE_DEVICES='0,1' in your environment before running")

# Set multiprocessing start method BEFORE any imports that use multiprocessing
if hasattr(multiprocessing, 'set_start_method'):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# NOW import torch - CUDA_VISIBLE_DEVICES is already set
import torch

# Verify CUDA setup
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"PyTorch sees {num_gpus} CUDA device(s)")
    if num_gpus < 2:
        print(f"ERROR: Only {num_gpus} GPU(s) visible, but tensor_parallel_size=2 required")
        print("This will cause initialization to fail.")
        print("Solution: Ensure CUDA_VISIBLE_DEVICES includes at least 2 GPUs")
else:
    print("ERROR: CUDA is not available")
    raise RuntimeError("CUDA is not available")

from vllm import LLM, SamplingParams

# Load model once (fast for repeated calls)
# For 5090x2 with 2 GPUs, use tensor_parallel_size=2
llm = LLM(
    "microsoft/phi-4", 
    tensor_parallel_size=2,
    enable_prefix_caching=True
)

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
