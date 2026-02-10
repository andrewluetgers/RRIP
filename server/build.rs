use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // mozjpeg and jpegli are mutually exclusive (both define jpeg_* symbols)
    #[cfg(all(feature = "mozjpeg", feature = "jpegli"))]
    compile_error!(
        "Features `mozjpeg` and `jpegli` are mutually exclusive — both provide libjpeg62 symbols."
    );

    let need_libjpeg_ffi = cfg!(feature = "mozjpeg") || cfg!(feature = "jpegli");

    if !need_libjpeg_ffi {
        return;
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    if cfg!(feature = "mozjpeg") {
        let lib_dir = env::var("MOZJPEG_LIB_DIR")
            .unwrap_or_else(|_| format!("{}/../vendor/mozjpeg/lib", manifest_dir));
        let include_dir = env::var("MOZJPEG_INCLUDE_DIR")
            .unwrap_or_else(|_| format!("{}/../vendor/mozjpeg/include", manifest_dir));

        compile_c_wrapper(&include_dir, &out_dir);

        // Use rustc-link-arg to force libraries onto the linker command line.
        // Static archive of our C wrapper:
        let wrapper_lib = out_dir.join("liblibjpeg_compress.a");
        println!("cargo:rustc-link-arg={}", wrapper_lib.display());
        // Static mozjpeg library:
        let jpeg_lib = format!("{}/libjpeg.a", lib_dir);
        println!("cargo:rustc-link-arg={}", jpeg_lib);
    }

    if cfg!(feature = "jpegli") {
        let lib_dir = env::var("JPEGLI_LIB_DIR")
            .unwrap_or_else(|_| format!("{}/../vendor/jpegli/lib", manifest_dir));
        let include_dir = env::var("JPEGLI_INCLUDE_DIR")
            .unwrap_or_else(|_| format!("{}/../vendor/jpegli/include/jpegli", manifest_dir));

        compile_c_wrapper(&include_dir, &out_dir);

        let wrapper_lib = out_dir.join("liblibjpeg_compress.a");
        println!("cargo:rustc-link-arg={}", wrapper_lib.display());
        println!("cargo:rustc-link-arg={}/libjpegli-static.a", lib_dir);
        println!("cargo:rustc-link-arg={}/libhwy.a", lib_dir);
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-lc++");
        } else {
            println!("cargo:rustc-link-arg=-lstdc++");
        }
    }

    println!("cargo:rerun-if-env-changed=MOZJPEG_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MOZJPEG_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=JPEGLI_LIB_DIR");
    println!("cargo:rerun-if-env-changed=JPEGLI_INCLUDE_DIR");
    println!("cargo:rerun-if-changed=csrc/libjpeg_compress.c");
}

/// Compile csrc/libjpeg_compress.c into a static library.
fn compile_c_wrapper(include_dir: &str, out_dir: &PathBuf) {
    let obj = out_dir.join("libjpeg_compress.o");
    let lib = out_dir.join("liblibjpeg_compress.a");

    let status = Command::new("cc")
        .args([
            "-c",
            "-O2",
            "-I",
            include_dir,
            "csrc/libjpeg_compress.c",
            "-o",
        ])
        .arg(&obj)
        .status()
        .expect("failed to run cc");
    assert!(status.success(), "cc failed to compile libjpeg_compress.c");

    let status = Command::new("ar")
        .args(["rcs"])
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");
}
