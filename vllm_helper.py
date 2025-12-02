import logging

# Set up logging BEFORE importing vLLM to capture all vLLM logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Enable detailed logging for vLLM modules
logging.getLogger("vllm").setLevel(logging.DEBUG)
logging.getLogger("vllm.engine").setLevel(logging.DEBUG)
logging.getLogger("vllm.worker").setLevel(logging.INFO)
logging.getLogger("vllm.scheduler").setLevel(logging.INFO)

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.info("Initializing vLLM with prefix caching enabled...")

from vllm import LLM, SamplingParams
# Load model once (fast for repeated calls)
llm = LLM("microsoft/phi-4", tensor_parallel_size=2, enable_prefix_caching=True)
logger.info("vLLM model loaded successfully")

# Debug: Print engine structure to find cache stats
def debug_engine_structure():
    """Debug function to explore engine structure and find cache stats."""
    if hasattr(llm, 'llm_engine'):
        engine = llm.llm_engine
        logger.debug(f"Engine type: {type(engine)}")
        logger.debug(f"Engine attributes: {[attr for attr in dir(engine) if not attr.startswith('_')]}")
        
        if hasattr(engine, 'stats'):
            stats = engine.stats
            logger.debug(f"Stats type: {type(stats)}")
            logger.debug(f"Stats attributes: {[attr for attr in dir(stats) if not attr.startswith('_')]}")
            logger.debug(f"Stats object: {stats}")
        
        if hasattr(engine, 'scheduler'):
            scheduler = engine.scheduler
            logger.debug(f"Scheduler type: {type(scheduler)}")
            if hasattr(scheduler, 'stats'):
                sched_stats = scheduler.stats
                logger.debug(f"Scheduler stats type: {type(sched_stats)}")
                logger.debug(f"Scheduler stats attributes: {[attr for attr in dir(sched_stats) if not attr.startswith('_')]}")

# Uncomment to debug engine structure
# debug_engine_structure()

params = SamplingParams(temperature=0.8, max_tokens=2000)

# Track prefix information to detect cache hits
_prefix_cache = {}  # Maps prefix_hash -> (first_tokens_processed, count)

def infer(prompt: str) -> dict:
    """Perform a single vLLM inference and return text, tokens_processed, and tokens_generated."""
    logger.debug(f"Starting inference, prompt length: {len(prompt)} characters")
    
    # Get stats before inference to track cache hits
    prefix_cache_hit_before = 0
    prefix_cache_miss_before = 0
    prefix_cache_hit_rate_before = 0.0
    
    try:
        if hasattr(llm, 'llm_engine') and llm.llm_engine is not None:
            engine = llm.llm_engine
            # Try to access stats from different possible locations
            if hasattr(engine, 'stats'):
                stats = engine.stats
                if hasattr(stats, 'num_prefix_cache_hits'):
                    prefix_cache_hit_before = stats.num_prefix_cache_hits
                if hasattr(stats, 'num_prefix_cache_misses'):
                    prefix_cache_miss_before = stats.num_prefix_cache_misses
                if hasattr(stats, 'gpu_prefix_cache_hit_rate'):
                    prefix_cache_hit_rate_before = stats.gpu_prefix_cache_hit_rate
            # Also check scheduler stats
            if hasattr(engine, 'scheduler') and hasattr(engine.scheduler, 'stats'):
                sched_stats = engine.scheduler.stats
                if hasattr(sched_stats, 'num_prefix_cache_hits'):
                    prefix_cache_hit_before = sched_stats.num_prefix_cache_hits
                if hasattr(sched_stats, 'num_prefix_cache_misses'):
                    prefix_cache_miss_before = sched_stats.num_prefix_cache_misses
    except Exception as e:
        logger.debug(f"Could not access prefix cache stats before: {e}")
    
    out = llm.generate([prompt], params)
    request_output = out[0]
    tokens_processed = len(request_output.prompt_token_ids)
    tokens_generated = len(request_output.outputs[0].token_ids)
    
    # Check request output for cache information
    cache_info_available = False
    try:
        # Check if request_output has any cache-related attributes
        if hasattr(request_output, 'metrics'):
            metrics = request_output.metrics
            logger.debug(f"Request metrics: {metrics}")
            cache_info_available = True
        # Check for prefix cache info in various possible locations
        for attr in ['prefix_cache_hit', 'cache_hit', 'prefix_hit', 'cached_tokens']:
            if hasattr(request_output, attr):
                value = getattr(request_output, attr)
                logger.info(f"Found cache attribute {attr}: {value}")
                cache_info_available = True
    except Exception as e:
        logger.debug(f"Could not check request output cache info: {e}")
    
    # Get stats after inference to see if cache was hit
    prefix_cache_hit_after = 0
    prefix_cache_miss_after = 0
    prefix_cache_hit_rate_after = 0.0
    cache_hit_this_request = False
    
    try:
        if hasattr(llm, 'llm_engine') and llm.llm_engine is not None:
            engine = llm.llm_engine
            if hasattr(engine, 'stats'):
                stats = engine.stats
                if hasattr(stats, 'num_prefix_cache_hits'):
                    prefix_cache_hit_after = stats.num_prefix_cache_hits
                if hasattr(stats, 'num_prefix_cache_misses'):
                    prefix_cache_miss_after = stats.num_prefix_cache_misses
                if hasattr(stats, 'gpu_prefix_cache_hit_rate'):
                    prefix_cache_hit_rate_after = stats.gpu_prefix_cache_hit_rate
            if hasattr(engine, 'scheduler') and hasattr(engine.scheduler, 'stats'):
                sched_stats = engine.scheduler.stats
                if hasattr(sched_stats, 'num_prefix_cache_hits'):
                    prefix_cache_hit_after = sched_stats.num_prefix_cache_hits
                if hasattr(sched_stats, 'num_prefix_cache_misses'):
                    prefix_cache_miss_after = sched_stats.num_prefix_cache_misses
            
            # Check if cache was hit for this request
            if prefix_cache_hit_after > prefix_cache_hit_before:
                cache_hit_this_request = True
    except Exception as e:
        logger.debug(f"Could not access prefix cache stats after: {e}")
    
    # Additional method: Track prefix by extracting system message
    # The system message should be the cacheable prefix
    import hashlib
    # Extract system message (everything up to <|im_start|>user)
    if "<|im_start|>system" in prompt and "<|im_start|>user" in prompt:
        system_end = prompt.find("<|im_start|>user")
        system_prefix = prompt[:system_end]
        prefix_hash = hashlib.md5(system_prefix.encode()).hexdigest()[:8]
        
        if prefix_hash in _prefix_cache:
            prev_tokens, count = _prefix_cache[prefix_hash]
            if tokens_processed < prev_tokens:
                cache_hit_this_request = True
                logger.info(f"✅ DETECTED CACHE HIT via token comparison! Previous: {prev_tokens}, Current: {tokens_processed}")
            _prefix_cache[prefix_hash] = (min(tokens_processed, prev_tokens), count + 1)
        else:
            _prefix_cache[prefix_hash] = (tokens_processed, 1)
            logger.debug(f"New prefix hash {prefix_hash}, tokens_processed={tokens_processed}")
    
    # Log cache information
    if cache_hit_this_request:
        logger.info(f"✅ PREFIX CACHE HIT! tokens_processed={tokens_processed}, tokens_generated={tokens_generated}")
    else:
        logger.info(f"❌ PREFIX CACHE MISS. tokens_processed={tokens_processed}, tokens_generated={tokens_generated}")
    
    if prefix_cache_hit_rate_after > 0:
        logger.info(f"Overall prefix cache hit rate: {prefix_cache_hit_rate_after:.2%}")
    
    return {
        "text": request_output.outputs[0].text,
        "tokens_processed": tokens_processed,
        "tokens_generated": tokens_generated,
        "prefix_cache_hit": cache_hit_this_request,
        "prefix_cache_hit_rate": prefix_cache_hit_rate_after
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
