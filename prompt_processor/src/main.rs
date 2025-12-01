use std::fs;
use std::path::Path;

fn main() {
    // Try to find prompt_files directory - check current dir first, then parent
    let prompt_files_dir = if Path::new("prompt_files").exists() {
        Path::new("prompt_files")
    } else if Path::new("../prompt_files").exists() {
        Path::new("../prompt_files")
    } else {
        eprintln!("Error: Could not find 'prompt_files' directory in current or parent directory");
        std::process::exit(1);
    };
    
    let output_dir = if Path::new("prompt_files_processed").exists() || Path::new("prompt_files").exists() {
        Path::new("prompt_files_processed")
    } else {
        Path::new("../prompt_files_processed")
    };

    // Create output directory if it doesn't exist
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }

    // Read all .txt files from prompt_files directory
    let entries = fs::read_dir(prompt_files_dir)
        .expect("Failed to read prompt_files directory");

    let mut processed_count = 0;
    let mut error_count = 0;

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Error reading directory entry: {}", e);
                error_count += 1;
                continue;
            }
        };

        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
            let file_name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown.txt");

            match process_file(&path, output_dir, file_name) {
                Ok(_) => {
                    println!("Processed: {}", file_name);
                    processed_count += 1;
                }
                Err(e) => {
                    eprintln!("Error processing {}: {}", file_name, e);
                    error_count += 1;
                }
            }
        }
    }

    println!("\nProcessing complete!");
    println!("Successfully processed: {} files", processed_count);
    if error_count > 0 {
        println!("Errors encountered: {} files", error_count);
    }
}

fn process_file(input_path: &Path, output_dir: &Path, file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Read the input file
    let content = fs::read_to_string(input_path)?;
    
    // Find all section markers
    let system_prompt_end = content.find("explain exactly why.\\n\\n");
    let req_doc_start = content.find("Requirement Document:");
    let device_output_start = content.find("Device Output:");
    let expected_output_start = content.find("Expected Output Format:");

    if system_prompt_end.is_none() || req_doc_start.is_none() || device_output_start.is_none() {
        return Err(format!("Missing required sections in file: {}", file_name).into());
    }

    let system_prompt_end = system_prompt_end.unwrap() + "explain exactly why.\\n\\n".len();
    let req_doc_start = req_doc_start.unwrap();
    let device_output_start = device_output_start.unwrap();
    
    // Extract system prompt (everything up to and including "explain exactly why.\n\n")
    let system_prompt = content[..system_prompt_end].trim();

    // Extract Requirement Document section
    let req_doc_end = device_output_start;
    let requirement_doc = content[req_doc_start..req_doc_end].trim();

    // Extract Device Output section (from Device Output to Expected Output Format, or to end if not found)
    let device_output_end = expected_output_start.unwrap_or(content.len());
    let device_output = content[device_output_start..device_output_end].trim();

    // Extract Expected Output Format section (if it exists, it comes after Device Output)
    let expected_output = if let Some(eof_start) = expected_output_start {
        content[eof_start..].trim()
    } else {
        ""
    };

    // Combine sections in the new order: System Prompt -> Expected Output -> Requirement Document -> Device Output
    let processed_content = if !expected_output.is_empty() {
        format!("{}\n\n{}\n\n{}\n\n{}", 
            system_prompt, 
            expected_output, 
            requirement_doc, 
            device_output)
    } else {
        format!("{}\n\n{}\n\n{}", 
            system_prompt, 
            requirement_doc, 
            device_output)
    };

    // Write to output file
    let output_path = output_dir.join(file_name);
    fs::write(&output_path, processed_content)?;

    Ok(())
}

