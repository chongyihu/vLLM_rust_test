use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use std::time::Instant;
use std::fs;
use std::env;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct PromptResult {
    prompt: String,
    response: String,
    time_ms: f64,
    tokens_processed: usize,
    tokens_generated: usize,
}

#[derive(Serialize, Deserialize)]
struct Results {
    results: Vec<PromptResult>,
    mean_time_ms: f64,
    mean_tokens_processed: f64,
    mean_tokens_generated: f64,
    total_prompts: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set environment variable to disable vLLM progress bar
    env::set_var("VLLM_NO_PROGRESS_BAR", "1");
    
    let prompts_quantity: usize = env::args()
        .nth(1)
        .ok_or("Please provide prompts_quantity as an argument(must be multiple of 50)")?
        .parse()
        .map_err(|_| "prompts_quantity must be a valid integer")?;
    if prompts_quantity % 50 != 0 {
        return Err("prompts_quantity must be a multiple of 50".into());
    }
    // Read test_prompts.json
    let json_content = fs::read_to_string("test_prompts_50.json")?;
    let json: serde_json::Value = serde_json::from_str(&json_content)?;
    let prompts = json["prompts"].as_array().unwrap();

    let mut all_results = Vec::new();
    let mut times = Vec::new();
    let mut tokens_processed_vec = Vec::new();
    let mut tokens_generated_vec = Vec::new();

    Python::with_gil(|py| -> PyResult<()> {
        // Add project root to sys.path
        let sys = py.import_bound("sys")?;
        let path_attr = sys.getattr("path")?;
        let path = path_attr.downcast::<PyList>()?;
        path.insert(0, ".")?; // insert current project root

        // Import your Python module
        let vllm = py.import_bound("vllm_helper")?;
        let infer_func = vllm.getattr("infer")?;

        // Iterate through prompts
        for i in 0..prompts_quantity / 50 {
            for prompt in prompts.iter() {
                let prompt_text = prompt.as_str().unwrap();
                
                // Format as "user: {prompt}\nassistant: "
                let formatted_prompt = format!("user: {}\nassistant: ", prompt_text);
                
                // Time the inference
                let start = Instant::now();
                let result_dict = infer_func.call1((formatted_prompt.as_str(),))?;
                let elapsed = start.elapsed();
                
                // Extract values from Python dictionary
                let dict = result_dict.downcast::<PyDict>()?;
                let response: String = dict.get_item("text")?.unwrap().extract()?;
                let tokens_processed: usize = dict.get_item("tokens_processed")?.unwrap().extract()?;
                let tokens_generated: usize = dict.get_item("tokens_generated")?.unwrap().extract()?;
                
                let time_ms = elapsed.as_millis() as f64;
                times.push(time_ms);
                tokens_processed_vec.push(tokens_processed as f64);
                tokens_generated_vec.push(tokens_generated as f64);
                
                all_results.push(PromptResult {
                    prompt: prompt_text.to_string(),
                    response,
                    time_ms,
                    tokens_processed,
                    tokens_generated,
                });
            }
        }

        // Cleanup: shutdown vLLM engine to release GPU memory
        let cleanup_func = vllm.getattr("cleanup")?;
        cleanup_func.call0()?;

        Ok(())
    }).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    // Calculate means
    let mean_time = times.iter().sum::<f64>() / times.len() as f64;
    let mean_tokens_processed = tokens_processed_vec.iter().sum::<f64>() / tokens_processed_vec.len() as f64;
    let mean_tokens_generated = tokens_generated_vec.iter().sum::<f64>() / tokens_generated_vec.len() as f64;
    
    // Create results structure
    let total_processed = all_results.len();
    let results = Results {
        results: all_results,
        mean_time_ms: mean_time,
        mean_tokens_processed,
        mean_tokens_generated,
        total_prompts: total_processed,
    };

    // Write to JSON file
    let json_output = serde_json::to_string_pretty(&results)?;
    fs::write(format!("result_rust_{}.json", prompts_quantity), json_output)?;

    println!("Results written to result_rust_{}.json", prompts_quantity);
    println!("Mean inference time: {:.2}ms", mean_time);
    println!("Mean tokens processed: {:.2}", mean_tokens_processed);
    println!("Mean tokens generated: {:.2}", mean_tokens_generated);
    println!("Total prompts processed: {}", total_processed);

    Ok(())
}
