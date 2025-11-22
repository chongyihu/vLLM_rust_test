use pyo3::prelude::*;
use pyo3::types::PyList;
use std::time::Instant;
use std::fs;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct PromptResult {
    prompt: String,
    response: String,
    time_ms: f64,
}

#[derive(Serialize, Deserialize)]
struct Results {
    results: Vec<PromptResult>,
    mean_time_ms: f64,
    total_prompts: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read test_prompts.json
    let json_content = fs::read_to_string("test_prompts.json")?;
    let json: serde_json::Value = serde_json::from_str(&json_content)?;
    let prompts = json["prompts"].as_array().unwrap();

    let mut all_results = Vec::new();
    let mut times = Vec::new();

    Python::with_gil(|py| -> PyResult<()> {
        // Add project root to sys.path
        let sys = py.import_bound("sys")?;
        let path: Bound<PyList> = sys.getattr("path")?.downcast()?;
        path.insert(0, ".")?; // insert current project root

        // Import your Python module
        let vllm = py.import_bound("vllm_helper")?;
        let infer_func = vllm.getattr("infer")?;

        // Iterate through prompts
        for prompt in prompts.iter() {
            let prompt_text = prompt.as_str().unwrap();
            
            // Format as "user: {prompt}\nassistant: "
            let formatted_prompt = format!("user: {}\nassistant: ", prompt_text);
            
            // Time the inference
            let start = Instant::now();
            let response: String = infer_func.call1((formatted_prompt.as_str(),))?.extract()?;
            let elapsed = start.elapsed();
            
            let time_ms = elapsed.as_millis() as f64;
            times.push(time_ms);
            
            all_results.push(PromptResult {
                prompt: prompt_text.to_string(),
                response,
                time_ms,
            });
        }

        Ok(())
    }).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    // Calculate mean time
    let mean_time = times.iter().sum::<f64>() / times.len() as f64;
    
    // Create results structure
    let results = Results {
        results: all_results,
        mean_time_ms: mean_time,
        total_prompts: prompts.len(),
    };

    // Write to JSON file
    let json_output = serde_json::to_string_pretty(&results)?;
    fs::write("result_rust.json", json_output)?;

    println!("Results written to result_rust.json");
    println!("Mean inference time: {:.2}ms", mean_time);
    println!("Total prompts processed: {}", prompts.len());

    Ok(())
}
