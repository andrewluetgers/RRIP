use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    // Compile CUDA kernels if nvcc is available
    let cuda_files = [
        "src/kernels/optl2.cu",
        "src/kernels/composite.cu",
        "src/kernels/residual.cu",
        "src/kernels/upsample.cu",
        "src/kernels/sharpen.cu",
    ];

    // GPU architecture: configurable via CUDA_ARCH env var
    // B200 = sm_100, H100 = sm_90, A100 = sm_80
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_100".to_string());

    // Find nvcc (prefer CUDA 12.8+ for Blackwell)
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let nvcc_available = std::process::Command::new(&nvcc)
        .arg("--version")
        .output()
        .is_ok();

    if nvcc_available {
        for cuda_file in &cuda_files {
            let stem = std::path::Path::new(cuda_file)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap();
            let ptx_path = out_dir.join(format!("{}.ptx", stem));

            let status = std::process::Command::new(&nvcc)
                .args([
                    "--ptx",
                    "-O3",
                    &format!("--gpu-architecture={}", arch),
                    "-o",
                ])
                .arg(&ptx_path)
                .arg(cuda_file)
                .status();

            match status {
                Ok(s) if s.success() => {
                    println!("cargo:warning=Compiled {} -> {}", cuda_file, ptx_path.display());
                }
                _ => {
                    println!("cargo:warning=Failed to compile {}, will use fallback", cuda_file);
                }
            }

            println!("cargo:rerun-if-changed={}", cuda_file);
        }
    } else {
        println!("cargo:warning=nvcc not found, CUDA kernels will not be compiled");
        println!("cargo:warning=Install CUDA Toolkit 12.6+ for GPU encoding support");
    }

    // Export the OUT_DIR so the crate can find PTX files at runtime
    println!("cargo:rustc-env=CUDA_PTX_DIR={}", out_dir.display());

    // Link nvJPEG (part of CUDA Toolkit)
    // CUDA_PATH can override the default search path
    let cuda_path = env::var("CUDA_PATH")
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-search=native={}/lib", cuda_path);
    println!("cargo:rustc-link-lib=dylib=nvjpeg");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
