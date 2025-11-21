use pyo3::prelude::*;
use pyo3::types::PyList;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        // Add current directory to Python path
        let sys = py.import_bound("sys")?;
        let path: &PyList = sys.getattr("path")?.downcast()?;
        path.insert(0, std::env::current_dir()?.to_str().unwrap())?;

        // Import your Python module
        let vllm = py.import_bound("vllm_helper")?;

        // Call the Python function
        let infer_func = vllm.getattr("infer")?;
        let response: String = infer_func
            .call1(("Hello from Rust!",))?
            .extract()?;

        println!("Model output: {}", response);

        Ok(())
    })
}
