use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::PathBuf;
use std::str::FromStr;

const KERNEL_FILES: [&str; 2] = ["gated_activation.cu", "gelu_erf_gate.cu"];

fn main() -> Result<()> {
    let num_cpus = std::env::var("RAYON_NUM_THREADS").map_or_else(
        |_| num_cpus::get_physical(),
        |s| usize::from_str(&s).unwrap(),
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus)
        .build_global()
        .unwrap();

    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed=kernels/{kernel_file}");
    }
    println!("cargo:rerun-if-env-changed=CANDLE_GATED_ACTIVATION_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CANDLE_NVCC_CCBIN");
    set_cuda_include_dir()?;

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("CANDLE_GATED_ACTIVATION_BUILD_DIR") {
        Err(_) => out_dir.clone(),
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            path.canonicalize().unwrap_or_else(|_| {
                panic!(
                    "Directory does not exist: {} (the current directory is {})",
                    path.display(),
                    std::env::current_dir()
                        .map(|dir| dir.display().to_string())
                        .unwrap_or_else(|_| "<unknown>".to_string())
                )
            })
        }
    };

    let ccbin_env = std::env::var("CANDLE_NVCC_CCBIN");
    let compute_cap = compute_cap()?;
    let out_file = build_dir.join("libgatedactivation.a");
    let stamp_file = build_dir.join("libgatedactivation.stamp");
    let build_stamp = format!(
        "KERNEL_BUILD_VERSION=4\nCUDA_COMPUTE_CAP={compute_cap}\nCANDLE_NVCC_CCBIN={}\n",
        ccbin_env.as_deref().unwrap_or("")
    );

    let kernel_dir = PathBuf::from("kernels");
    let cu_files: Vec<_> = KERNEL_FILES
        .iter()
        .map(|f| {
            let mut obj_file = out_dir.join(f);
            obj_file.set_extension("o");
            (kernel_dir.join(f), obj_file)
        })
        .collect();

    let out_modified: Result<_, _> = out_file.metadata().and_then(|m| m.modified());
    let should_compile = if out_file.exists() {
        std::fs::read_to_string(&stamp_file).ok().as_deref() != Some(build_stamp.as_str())
            || kernel_dir
                .read_dir()
                .expect("kernels folder should exist")
                .any(|entry| {
                    if let (Ok(entry), Ok(out_modified)) = (entry, &out_modified) {
                        let in_modified = entry.metadata().unwrap().modified().unwrap();
                        in_modified.duration_since(*out_modified).is_ok()
                    } else {
                        true
                    }
                })
    } else {
        true
    };

    if should_compile {
        cu_files
            .par_iter()
            .map(|(cu_file, obj_file)| {
                let mut command = std::process::Command::new("nvcc");
                command
                    .arg("-std=c++17")
                    .arg("-O3")
                    .arg("-U__CUDA_NO_HALF_OPERATORS__")
                    .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
                    .arg("-U__CUDA_NO_BFLOAT16_OPERATORS__")
                    .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
                    .arg("-U__CUDA_NO_BFLOAT162_OPERATORS__")
                    .arg("-U__CUDA_NO_BFLOAT162_CONVERSIONS__")
                    .arg(format!("--gpu-architecture=sm_{compute_cap}"))
                    .arg("-c")
                    .args(["-o", obj_file.to_str().unwrap()])
                    .arg("--expt-relaxed-constexpr")
                    .arg("--expt-extended-lambda");
                if cu_file.file_name().and_then(|name| name.to_str()) == Some("gated_activation.cu")
                {
                    command.arg("--use_fast_math");
                }
                command.arg("--verbose");
                if let Ok(ccbin_path) = &ccbin_env {
                    command
                        .arg("-allow-unsupported-compiler")
                        .args(["-ccbin", ccbin_path]);
                }
                command.arg(cu_file);
                let output = command
                    .spawn()
                    .context("failed spawning nvcc")?
                    .wait_with_output()?;
                if !output.status.success() {
                    anyhow::bail!(
                        "nvcc error while compiling {:?}\n\n# stdout\n{}\n\n# stderr\n{}",
                        command,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )
                }
                Ok(())
            })
            .collect::<Result<()>>()?;

        let obj_files = cu_files.iter().map(|c| c.1.clone()).collect::<Vec<_>>();
        let mut command = std::process::Command::new("nvcc");
        command
            .arg("--lib")
            .args(["-o", out_file.to_str().unwrap()])
            .args(obj_files);
        let output = command
            .spawn()
            .context("failed spawning nvcc")?
            .wait_with_output()?;
        if !output.status.success() {
            anyhow::bail!(
                "nvcc error while linking {:?}\n\n# stdout\n{}\n\n# stderr\n{}",
                command,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        }
        std::fs::write(&stamp_file, build_stamp).with_context(|| {
            format!(
                "write gated-activation build stamp {}",
                stamp_file.display()
            )
        })?;
    }

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=gatedactivation");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}

fn set_cuda_include_dir() -> Result<()> {
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .map(Into::<PathBuf>::into);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];
    let root = env_vars
        .chain(roots.into_iter().map(Into::<PathBuf>::into))
        .find(|path| path.join("include").join("cuda.h").is_file())
        .context("cannot find include/cuda.h")?;
    println!(
        "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
        root.join("include").display()
    );
    Ok(())
}

fn compute_cap() -> Result<usize> {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    let compute_cap = if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap_str}");
        compute_cap_str
            .parse::<usize>()
            .context("could not parse CUDA_COMPUTE_CAP")?
    } else {
        let out = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=compute_cap")
            .arg("--format=csv")
            .output()
            .context("`nvidia-smi` failed. Ensure CUDA is installed and nvidia-smi is in PATH.")?;
        let out = std::str::from_utf8(&out.stdout).context("stdout is not utf8")?;
        let mut lines = out.lines();
        assert_eq!(
            lines.next().context("missing line in stdout")?,
            "compute_cap"
        );
        let cap = lines
            .next()
            .context("missing line in stdout")?
            .replace('.', "");
        let cap = cap
            .parse::<usize>()
            .with_context(|| format!("cannot parse as int {cap}"))?;
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={cap}");
        cap
    };

    let out = std::process::Command::new("nvcc")
        .arg("--list-gpu-code")
        .output()
        .context("`nvcc` failed. Ensure CUDA is installed and nvcc is in PATH.")?;
    let out = std::str::from_utf8(&out.stdout).context("nvcc stdout is not utf8")?;
    let mut supported_nvcc_codes = out
        .lines()
        .filter_map(|line| {
            let parts = line.split('_').collect::<Vec<_>>();
            if !parts.is_empty() && parts.contains(&"sm") {
                parts.get(1).and_then(|value| value.parse::<usize>().ok())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    supported_nvcc_codes.sort();
    let max_nvcc_code = *supported_nvcc_codes
        .last()
        .context("nvcc did not report any sm targets")?;

    if !supported_nvcc_codes.contains(&compute_cap) {
        anyhow::bail!(
            "nvcc cannot target gpu arch {compute_cap}. Available nvcc targets are {supported_nvcc_codes:?}."
        );
    }
    if compute_cap > max_nvcc_code {
        anyhow::bail!(
            "CUDA compute cap {compute_cap} is higher than highest nvcc gpu code {max_nvcc_code}"
        );
    }

    Ok(compute_cap)
}
