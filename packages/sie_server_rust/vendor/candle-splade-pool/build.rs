use anyhow::{Context, Result};
use std::path::PathBuf;

const KERNEL_FILE: &str = "splade_pool.cu";

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/{KERNEL_FILE}");
    println!("cargo:rerun-if-env-changed=CANDLE_SPLADE_POOL_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CANDLE_NVCC_CCBIN");
    set_cuda_include_dir()?;

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("CANDLE_SPLADE_POOL_BUILD_DIR") {
        Ok(build_dir) => PathBuf::from(build_dir)
            .canonicalize()
            .context("CANDLE_SPLADE_POOL_BUILD_DIR does not exist")?,
        Err(_) => out_dir.clone(),
    };
    let ccbin = std::env::var("CANDLE_NVCC_CCBIN").ok();
    let compute_cap = compute_cap()?;
    let kernel_path = PathBuf::from("kernels").join(KERNEL_FILE);
    let object_path = out_dir.join("splade_pool.o");
    let library_path = build_dir.join("libspladepool.a");
    let stamp_path = build_dir.join("libspladepool.stamp");
    let build_stamp = format!(
        "KERNEL_BUILD_VERSION=1\nCUDA_COMPUTE_CAP={compute_cap}\nCANDLE_NVCC_CCBIN={}\n",
        ccbin.as_deref().unwrap_or("")
    );

    let library_modified = library_path.metadata().and_then(|meta| meta.modified());
    let kernel_is_newer = match (kernel_path.metadata(), library_modified) {
        (Ok(kernel_meta), Ok(library_modified)) => kernel_meta
            .modified()
            .map(|modified| modified > library_modified)
            .unwrap_or(true),
        _ => true,
    };
    let should_compile = !library_path.exists()
        || kernel_is_newer
        || std::fs::read_to_string(&stamp_path).ok().as_deref() != Some(build_stamp.as_str());

    if should_compile {
        let mut compile = std::process::Command::new("nvcc");
        compile
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg(format!("--gpu-architecture=sm_{compute_cap}"))
            .arg("-c")
            .args(["-o", object_path.to_str().context("non-UTF8 object path")?])
            .arg("--verbose");
        if let Some(ccbin) = ccbin.as_deref() {
            compile
                .arg("-allow-unsupported-compiler")
                .args(["-ccbin", ccbin]);
        }
        compile.arg(&kernel_path);
        run(&mut compile, "compile SPLADE pooling CUDA kernel")?;

        let mut archive = std::process::Command::new("nvcc");
        archive
            .arg("--lib")
            .args([
                "-o",
                library_path.to_str().context("non-UTF8 library path")?,
            ])
            .arg(&object_path);
        run(&mut archive, "archive SPLADE pooling CUDA kernel")?;
        std::fs::write(&stamp_path, build_stamp).with_context(|| {
            format!("write SPLADE pooling build stamp {}", stamp_path.display())
        })?;
    }

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=spladepool");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    Ok(())
}

fn run(command: &mut std::process::Command, action: &str) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("failed to {action}"))?;
    if output.status.success() {
        return Ok(());
    }
    anyhow::bail!(
        "failed to {action}: {command:?}\n\n# stdout\n{}\n\n# stderr\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn set_cuda_include_dir() -> Result<()> {
    let env_roots = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ]
    .into_iter()
    .filter_map(|name| std::env::var(name).ok())
    .map(PathBuf::from);
    let standard_roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ]
    .into_iter()
    .map(PathBuf::from);
    let root = env_roots
        .chain(standard_roots)
        .find(|path| path.join("include/cuda.h").is_file())
        .context("cannot find include/cuda.h")?;
    println!(
        "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
        root.join("include").display()
    );
    Ok(())
}

fn compute_cap() -> Result<usize> {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    let compute_cap = if let Ok(value) = std::env::var("CUDA_COMPUTE_CAP") {
        value
            .parse::<usize>()
            .context("could not parse CUDA_COMPUTE_CAP")?
    } else {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .context("nvidia-smi failed; set CUDA_COMPUTE_CAP explicitly")?;
        std::str::from_utf8(&output.stdout)
            .context("nvidia-smi output is not UTF-8")?
            .lines()
            .next()
            .context("nvidia-smi returned no compute capability")?
            .trim()
            .replace('.', "")
            .parse::<usize>()
            .context("could not parse nvidia-smi compute capability")?
    };

    let output = std::process::Command::new("nvcc")
        .arg("--list-gpu-code")
        .output()
        .context("nvcc failed; ensure the CUDA toolkit is installed")?;
    let supported = std::str::from_utf8(&output.stdout)
        .context("nvcc output is not UTF-8")?
        .lines()
        .filter_map(|line| {
            let parts = line.split('_').collect::<Vec<_>>();
            if parts.contains(&"sm") {
                parts.get(1).and_then(|code| code.parse().ok())
            } else {
                None
            }
        })
        .collect::<Vec<usize>>();
    if !supported.contains(&compute_cap) {
        anyhow::bail!(
            "nvcc cannot target GPU architecture {compute_cap}; available targets: {supported:?}"
        )
    }
    println!("cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap}");
    Ok(compute_cap)
}
