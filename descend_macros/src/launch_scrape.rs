//! Reading a launch's shared-memory byte count back out of Descend's generated CUDA.
//!
//! The pipeline does not do this: it sizes shared memory from the parameter types, in
//! `check::shared_mem_bytes`. This module exists so a test can confirm the two agree,
//! which is the contract that lets the typed answer stand alone.

use crate::const_expr::eval_const_int;

pub fn shared_bytes_expr(host_fn: &str) -> Result<String, String> {
    let open = host_fn
        .find("<<<")
        .ok_or_else(|| "no kernel launch found in the generated host code".to_string())?;
    let rest = &host_fn[open + 3..];
    let close = rest
        .find(">>>")
        .ok_or_else(|| "the generated kernel launch is not terminated".to_string())?;
    let inside = &rest[..close];

    let args = split_top_level(inside);
    match args.as_slice() {
        [_grid, _block, bytes] => Ok(bytes.split_whitespace().collect()),
        other => Err(format!(
            "expected 3 launch arguments, found {} in `<<<{}>>>`",
            other.len(),
            inside.trim()
        )),
    }
}

fn split_top_level(args: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, c) in args.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&args[start..i]);
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&args[start..]);
    parts
}

pub fn scraped_shared_mem_bytes(host_fn: &str) -> Result<u64, String> {
    eval_const_int(&shared_bytes_expr(host_fn)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SGEMM_LAUNCH: &str = "__host__ auto dummy_main() -> void {
  sgemm_gpu<<<dim3(8, 16, 1), dim3(32, 8, 1),
              ((0 + (4 * ((1 * (32 * 2)) * 8))) + (4 * ((1 * (32 * 4)) * 8)))>>>(
      a, b, c);
}";

    #[test]
    fn reads_the_count_across_wrapped_lines() {
        assert_eq!(
            shared_bytes_expr(SGEMM_LAUNCH).unwrap(),
            "((0+(4*((1*(32*2))*8)))+(4*((1*(32*4))*8)))"
        );
        assert_eq!(scraped_shared_mem_bytes(SGEMM_LAUNCH).unwrap(), 6144);
    }

    #[test]
    fn a_kernel_without_shared_parameters_requests_nothing() {
        let launch = "gpu_vec_add<<<dim3(64, 1, 1), dim3(256, 1, 1), 0>>>(a, b);";
        assert_eq!(scraped_shared_mem_bytes(launch).unwrap(), 0);
    }

    #[test]
    fn dim3_commas_do_not_split_the_argument_list() {
        let launch = "k<<<dim3(1, 2, 3), dim3(4, 5, 6), (4 * 64)>>>(x);";
        assert_eq!(scraped_shared_mem_bytes(launch).unwrap(), 256);
    }

    #[test]
    fn an_unresolved_size_is_an_error() {
        let launch = "k<<<dim3(1, 1, 1), dim3(1, 1, 1), (4 * bs)>>>(x);";
        let error = scraped_shared_mem_bytes(launch).unwrap_err();
        assert!(
            error.contains("not a constant integer expression"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_missing_launch_is_an_error() {
        assert!(shared_bytes_expr("__host__ auto dummy_main() -> void {}").is_err());
    }

    /// The contract that lets `check::shared_mem_bytes` stand alone: the count the
    /// parameter types give is the count Descend's own code generator emits.
    ///
    /// This runs the instantiation pipeline up to the generated CUDA, so it needs no
    /// nvcc and no GPU.
    const SHARED_KERNEL: &str = r#"
fn gpu_reduce<gs: nat, bs: nat, r1: prv, r2: prv, r3: prv>(
        vec: &r1 shrd gpu.global [i32; gs*bs],
        out_vec: &r2 uniq gpu.global [i32; gs],
        [grid.forall(X)] tmp: &r3 uniq gpu.shared [i32; bs]
    ) -[grid: gpu.grid<X<gs>, X<bs>>]-> () {
        let block_group = &shrd (*vec).to_view.grp::<bs>;
        let view_out = &uniq (*out_vec).to_view.grp::<1>;
        sched block in grid {
            let ib = &shrd (*block_group)[[block]];
            let out_elem = &uniq (*view_out)[[block]];
            let tmp_view = &uniq (*tmp).to_view;
            sched thread in block {
                let inp = &shrd (*ib)[[thread]];
                let tmp_in = &uniq (*tmp_view)[[thread]];
                *tmp_in = *inp
            };

            for k in halved_range(bs/2) {
                sync(block);
                let split_tmp_view = &uniq (*tmp_view)[0..2*k];
                let active_half0 = &uniq (*split_tmp_view)[0..k];
                let active_half1 = &uniq (*split_tmp_view)[k..2*k];
                indep(X) k block {
                    active =>
                        sched thread in active {
                            let a = &uniq (*active_half0)[[thread]];
                            let b = &shrd (*active_half1)[[thread]];
                            *a = *a + *b
                        },
                    inactive => { () }
                }
            };

            sync(block);
            indep(X) 1 block {
                single_thread =>
                    sched thread in single_thread {
                        let out = &uniq (*out_elem)[[thread]];
                        let tmp_val = &shrd (*tmp_view)[0..1][[thread]];
                        *out = *tmp_val
                    },
                rest => { () }
            }
        }
    }
"#;

    #[test]
    fn the_typed_count_agrees_with_the_generated_launch() {
        use std::collections::HashMap;
        use std::str::FromStr;

        let bindings = HashMap::from([
            ("gs".to_string(), "4".to_string()),
            ("bs".to_string(), "64".to_string()),
        ]);
        let specialized = crate::specialize::specialize_kernel(
            proc_macro2::TokenStream::from_str(SHARED_KERNEL).expect("fixture must tokenize"),
            "gpu_reduce",
            &bindings,
        )
        .expect("fixture must specialize");

        let source = crate::source::DescendSource::from_token_stream(specialized);
        let code = descend::parser::SourceCode::new(source.text.clone());
        let unit = crate::source::parse_descend_source(&code).expect("fixture must parse");
        let kernel = crate::kernel::Kernel::extract(&unit, "gpu_reduce", &source)
            .expect("fixture must extract");

        let typed = crate::check::shared_mem_bytes(&kernel).expect("typed count");
        assert_eq!(typed, 64 * 4, "a shared [i32; 64] tile is 256 bytes");

        let program = format!(
            "{}\n{}",
            crate::synthesize::synthesize_main_fn(&kernel).expect("main"),
            source.text
        );
        let cpp = crate::translate::translate(&program, "gpu_reduce").expect("fixture must build");

        assert_eq!(
            scraped_shared_mem_bytes(&cpp.host_fn),
            Ok(typed),
            "the generated launch disagrees with the parameter types:\n{}",
            cpp.host_fn
        );
    }

    #[test]
    fn a_launch_with_two_arguments_is_an_error() {
        let launch = "k<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(x);";
        let error = shared_bytes_expr(launch).unwrap_err();
        assert!(error.contains("expected 3 launch arguments"), "{error}");
    }
}
