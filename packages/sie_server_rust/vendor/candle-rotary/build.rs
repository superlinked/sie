// Build script to run nvcc and generate the C glue code for launching the rotary kernel.
// The CUDA build time can be long, so set CANDLE_ROTARY_BUILD_DIR to reuse compiled
// artifacts across builds.
use anyhow::{Context, Result};
use std::path::PathBuf;

const KERNEL_FILES: [&str; 1] = ["kernels/rotary.cu"];

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed={kernel_file}");
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("CANDLE_ROTARY_BUILD_DIR") {
        Err(_) =>
        {
            #[allow(clippy::redundant_clone)]
            out_dir.clone()
        }
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            let current_dir = std::env::current_dir()?;
            path.canonicalize().unwrap_or_else(|_| {
                panic!(
                    "Directory does not exist: {} (the current directory is {})",
                    path.display(),
                    current_dir.display()
                )
            })
        }
    };

    let kernels: Vec<_> = KERNEL_FILES.iter().collect();
    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(kernels)
        .out_dir(build_dir.clone())
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--ptxas-options=-v")
        .arg("--verbose");

    let out_file = build_dir.join("librotary.a");
    builder.build_lib(out_file);

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=rotary");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}
