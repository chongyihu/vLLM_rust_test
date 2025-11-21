use pyo3::prelude::*;
use pyo3::types::PyList;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        // Add project root to sys.path
        let sys = py.import("sys")?;
        let path: &PyList = sys.getattr("path")?.downcast()?;
        path.insert(0, ".")?; // insert current project root

        // Import your Python module
        let vllm = PyModule::import(py, "vllm_helper")?;

        let infer_func = vllm.getattr("infer")?;
        let response: String = infer_func.call1(("Hello from Rust!",))?.extract()?;

        println!("Model output: {}", response);
        Ok(())
    })
}
