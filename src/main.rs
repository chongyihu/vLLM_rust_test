use pyo3::prelude::*;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        // Import your Python module
        let vllm = PyModule::import(py, "my_vllm")?;

        // Call the Python function
        let infer_func = vllm.getattr("infer")?;
        let response: String = infer_func
            .call1(("Hello from Rust!",))?
            .extract()?;

        println!("Model output: {}", response);

        Ok(())
    })
}
