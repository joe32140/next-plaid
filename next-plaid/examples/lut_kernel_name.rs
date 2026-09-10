//! Report which residual-LUT kernel this CPU will actually dispatch, and fail
//! if that is the scalar fallback.
//!
//! `/proc/cpuinfo` is the wrong oracle for this: Linux spells the AVX-512
//! VNNI bit `avx512_vnni` while the Rust feature name is `avx512vnni`, so a
//! grep for one silently never matches, and a job can report "no AVX-512"
//! on a CPU whose dispatch is using it. This asks the same detection the
//! kernel dispatch asks.
//!
//! Usage: cargo run -p next-plaid --release --example lut_kernel_name
//! Env: ALLOW_SCALAR=1 to report instead of failing (for a runner that
//!      legitimately has no SIMD path).
fn main() {
    println!("target_arch = {}", std::env::consts::ARCH);
    #[cfg(target_arch = "x86_64")]
    {
        // `is_x86_feature_detected!` takes a literal, so these are spelled out.
        for (name, on) in [
            ("avx2", std::arch::is_x86_feature_detected!("avx2")),
            ("avx512f", std::arch::is_x86_feature_detected!("avx512f")),
            ("avx512bw", std::arch::is_x86_feature_detected!("avx512bw")),
            (
                "avx512vnni",
                std::arch::is_x86_feature_detected!("avx512vnni"),
            ),
            ("avxvnni", std::arch::is_x86_feature_detected!("avxvnni")),
        ] {
            println!("  feature {name}: {on}");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        for (name, on) in [
            ("neon", std::arch::is_aarch64_feature_detected!("neon")),
            (
                "dotprod",
                std::arch::is_aarch64_feature_detected!("dotprod"),
            ),
            ("i8mm", std::arch::is_aarch64_feature_detected!("i8mm")),
        ] {
            println!("  feature {name}: {on}");
        }
    }

    let mut any = false;
    for dim in [48usize, 128, 256] {
        let name = next_plaid::residual_lut::active_kernel_name(dim, true);
        let ok = next_plaid::residual_lut::simd_dispatch_available(dim, true);
        println!("  dim {dim}: kernel = {name}, simd = {ok}");
        if dim == 128 {
            any = ok;
        }
    }

    if !any {
        let allow = std::env::var("ALLOW_SCALAR")
            .map(|v| v == "1")
            .unwrap_or(false);
        if allow {
            println!("no SIMD kernel here; ALLOW_SCALAR=1 so this is not a failure");
        } else {
            eprintln!(
                "error: dim 128 dispatches to the scalar fallback on this CPU, so a kernel \
                 test here would pass without executing a kernel"
            );
            std::process::exit(1);
        }
    }
}
