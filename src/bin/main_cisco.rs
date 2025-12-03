use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use std::time::Instant;
use std::fs;
use std::path::Path;
use std::env;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct CiscoResult {
    filename: String,
    processing_time: f64,
    tokens_processed: usize,
    tokens_generated: usize,
}

#[derive(Serialize, Deserialize)]
struct CiscoResults {
    results: Vec<CiscoResult>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set environment variable to disable vLLM progress bar
    unsafe {
        env::set_var("VLLM_NO_PROGRESS_BAR", "1");
    }
    
    let prompt_files_dir = "prompt_021_processed";
    
    // Read all .txt files from prompt_files directory
    let mut files: Vec<String> = Vec::new();
    let dir = fs::read_dir(prompt_files_dir)?;
    
    for entry in dir {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.contains("cisco") {
                files.push(filename.to_string());
                }
            }
        }
    }
    
    
    // Sort files for consistent processing order
    files.sort();
    let mut sys_prompt = fs::read_to_string(format!("{}/sys_prompt.txt", prompt_files_dir))?;
    let mut all_results = Vec::new();


    Python::with_gil(|py| -> PyResult<()> {
        // Add project root to sys.path
        let sys = py.import_bound("sys")?;
        let path_attr = sys.getattr("path")?;
        let path = path_attr.downcast::<PyList>()?;
        path.insert(0, ".")?; // insert current project root

        // Import your Python module
        let vllm = py.import_bound("vllm_helper")?;
        let infer_func = vllm.getattr("infer")?;

        // Process each file
        let mut count = 0;
        for filename in &files {
            count += 1;
            let file_path = Path::new(prompt_files_dir).join(filename);
            
            // Read the prompt from file
            let prompt_content = fs::read_to_string(&file_path)?;
            
            // Format as "user: {prompt}\nassistant: "
            sys_prompt.insert_str(40, &count.to_string());
            let formatted_prompt = format!("{}\n{}",sys_prompt, prompt_content);
            
            // Time the inference
            let start = Instant::now();
            let result_dict = infer_func.call1((formatted_prompt.as_str(),))?;
            let elapsed = start.elapsed();
            
            // Extract values from Python dictionary
            let dict = result_dict.downcast::<PyDict>()?;
            let response: String = dict.get_item("text")?.unwrap().extract()?;
            let tokens_processed: usize = dict.get_item("tokens_processed")?.unwrap().extract()?;
            let tokens_generated: usize = dict.get_item("tokens_generated")?.unwrap().extract()?;
            
            let processing_time = elapsed.as_millis() as f64;
            
            // Print the model output
            println!("\n=== {} ===", filename);
            println!("Processing time: {:.2}ms", processing_time);
            println!("Tokens processed: {}, Tokens generated: {}", tokens_processed, tokens_generated);
            println!("Model output:\n{}", response);
            println!("{}\n", "=".repeat(80));
            
            all_results.push(CiscoResult {
                filename: filename.clone(),
                processing_time,
                tokens_processed,
                tokens_generated,
            });
        }

        // Cleanup: shutdown vLLM engine to release GPU memory
        let cleanup_func = vllm.getattr("cleanup")?;
        cleanup_func.call0()?;

        Ok(())
    }).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    // Create results structure
    let results = CiscoResults {
        results: all_results,
    };

    // Write to JSON file
    let json_output = serde_json::to_string_pretty(&results)?;
    fs::write("cisco_test_result.json", json_output)?;

    println!("\nResults written to cisco_test_result.json");
    println!("Total files processed: {}", results.results.len());

    Ok(())
}
