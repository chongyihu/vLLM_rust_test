use std::fs;
use std::path::Path;

fn main() {
    // Try to find prompt_021 directory - check current dir first, then parent
    let prompt_files_dir = if Path::new("prompt_021").exists() {
        Path::new("prompt_021")
    } else if Path::new("../prompt_021").exists() {
        Path::new("../prompt_021")
    } else {
        eprintln!("Error: Could not find 'prompt_021' directory in current or parent directory");
        std::process::exit(1);
    };
    
    let output_dir = if Path::new("prompt_021_processed").exists() || Path::new("prompt_021").exists() {
        Path::new("prompt_021_processed")
    } else {
        Path::new("../prompt_021_processed")
    };

    // Create output directory if it doesn't exist
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }

    // Read all .txt files from prompt_021 directory
    let entries = fs::read_dir(prompt_files_dir)
        .expect("Failed to read prompt_021 directory");

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
    // The sysprompt is everything from the beginning up to and including "explain exactly why.\n\n"
    let sysprompt_end_marker = "explain exactly why.\\n\\n";
    let sysprompt_end = content.find(sysprompt_end_marker);
    let req_doc_start = content.find("Requirement Document:");
    let device_output_start = content.find("Device Output:");
    let expected_output_start = content.find("Expected Output Format:");

    if sysprompt_end.is_none() || req_doc_start.is_none() || device_output_start.is_none() {
        return Err(format!("Missing required sections in file: {}", file_name).into());
    }

    let sysprompt_end_pos = sysprompt_end.unwrap() + sysprompt_end_marker.len();
    let req_doc_start_pos = req_doc_start.unwrap();
    let device_output_start_pos = device_output_start.unwrap();
    
    // Extract system prompt (from beginning to "explain exactly why.\n\n")
    let sysprompt = content[..sysprompt_end_pos].trim();

    // Extract Requirement Document section
    let req_doc_end = device_output_start_pos;
    let requirement_doc = content[req_doc_start_pos..req_doc_end].trim();

    // Extract Device Output section (from Device Output to Expected Output Format, or to end if not found)
    let device_output_end = expected_output_start.unwrap_or(content.len());
    let device_output = content[device_output_start_pos..device_output_end].trim();

    // Extract Expected Output Format section (if it exists, it comes after Device Output)
    let expected_output = if let Some(eof_start) = expected_output_start {
        content[eof_start..].trim()
    } else {
        ""
    };

    // Format according to the template:
    // <|im_start|>system<|im_sep|>
    // Sysprompt, Requirement Document, Expected Output Format
    // <|im_start|>user<|im_sep|>
    // Device Output<|im_end|>
    // <|im_start|>assistant<|im_sep|>
    let processed_content = 
        format!("<|im_start|>system<|im_sep|>\n{}\n{}\n{}\n<|im_end|>\n<|im_start|>user<|im_sep|>\n{}<|im_end|>\n<|im_start|>assistant<|im_sep|>", 
            sysprompt, 
            requirement_doc,
            expected_output,
            device_output);

    // Write to output file
    let output_path = output_dir.join(file_name);
    fs::write(&output_path, processed_content)?;

    Ok(())
}

