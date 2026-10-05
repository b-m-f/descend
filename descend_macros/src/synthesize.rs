//! The Descend `main` a kernel is type-checked against.
//!
//! A kernel on its own is not a program, so the macro writes the smallest `main` that
//! calls it: allocate a GPU buffer per host parameter, copy, launch. Descend's type
//! checker then runs over the two together, and it is what proves the kernel's
//! parameters agree with the buffers the host will hand it.
//!
//! That `main` is never compiled -- `translate::generate_cpp_source` renames it and
//! removes it -- but it is the subject of every diagnostic `capture` reports, so its
//! parameter names are the names a reader is shown.

use crate::diag::MacroError;
use crate::kernel::{Kernel, KernelParameter};
use crate::spell::format_dimension;
use std::collections::HashSet;

/// The identifiers Descend's parser refuses to let anything else be called.
///
/// This mirrors the `keyword()` rule in `src/parser/mod.rs:964`, which is a private peg
/// rule and so cannot be imported. The duplication is checked rather than assumed:
/// `tests::the_keyword_list_matches_the_parser_rule` reads the rule back out of the
/// parser source and fails if the two have drifted.
const DESCEND_KEYWORDS: &[&str] = &[
    // `keyword()` proper.
    "crate",
    "super",
    "self",
    "Self",
    "const",
    "mut",
    "uniq",
    "shrd",
    "indep",
    "in",
    "to_thread_grp",
    "to_warps",
    "to",
    "with",
    "f32",
    "f64",
    "i32",
    "u8",
    "u32",
    "u64",
    "bool",
    "AtomicU32",
    "Gpu",
    "nat",
    "mem",
    "ty",
    "prv",
    "own",
    "let",
    "letprov",
    "if",
    "else",
    "sched",
    "for",
    "while",
    "fn",
    "split_exec",
    "split",
    "cpu.mem",
    "gpu.global",
    "gpu.shared",
    "sync",
    "struct",
    "unsafe",
    "cpu.thread",
    "gpu.grid",
    "gpu.block",
    "gpu.global_threads",
    "gpu.block_grp",
    "gpu.thread_grp",
    "gpu.thread",
    "any",
    // `view_name()`, which `keyword()` includes.
    "to_view",
    "grp",
    "map",
    "transp",
    "join",
    "rev",
];

/// What the synthesized `main` binds besides its parameters.
const RESERVED_LOCALS: &[&str] = &["gpu", "main"];

/// The names the synthesized `main` gives the kernel's parameters.
///
/// Descend's type checker reports against that `main`, which is the entire reason
/// `capture` exists, so a diagnostic that says `argument2` sends the reader looking for
/// a name they never wrote. Using the kernel's own parameter name makes it say
/// `b_array`.
///
/// A Descend parameter name is not checked against Descend's keywords, and two
/// parameters can produce the same device-copy local, so the names are only usable when
/// nothing collides. Where something does, the numbered names stand in for all of them:
/// a less readable diagnostic is a poor outcome, but a `main` that does not parse is
/// worse, and a mixture of the two schemes would be harder to follow than either.
fn main_parameter_names(kernel: &Kernel) -> Vec<String> {
    let numbered = || {
        (0..kernel.parameters.len())
            .map(|index| format!("argument{index}"))
            .collect()
    };

    let mut bound: HashSet<String> = RESERVED_LOCALS
        .iter()
        .map(|reserved| (*reserved).to_string())
        .collect();

    for parameter in &kernel.parameters {
        let name = &parameter.name;
        if DESCEND_KEYWORDS.contains(&name.as_str()) || !bound.insert(name.clone()) {
            return numbered();
        }
        // The device copy a reference parameter is allocated into. Two parameters can
        // ask for the same local: `x` and `x_gpu` both want `x_gpu`, whichever order
        // they are declared in, and `insert` reports the second as already present.
        if !bound.insert(device_copy_name(name)) {
            return numbered();
        }
    }

    kernel
        .parameters
        .iter()
        .map(|parameter| parameter.name.clone())
        .collect()
}

/// The local a reference parameter's device copy is bound to.
fn device_copy_name(parameter_name: &str) -> String {
    format!("{parameter_name}_gpu")
}

pub fn synthesize_main_function(kernel: &Kernel) -> Result<String, MacroError> {
    let names = main_parameter_names(kernel);

    let mut main_parameters = Vec::new();
    let mut main_bindings = Vec::new();
    let mut main_arguments = Vec::new();
    let mut shared_memory_configurations = Vec::new();
    let mut borrows = false;

    for (parameter, name) in kernel.parameters.iter().zip(&names) {
        if parameter.is_shared_memory {
            shared_memory_configurations.push(parameter.host.descend());
            continue;
        }

        if parameter.is_by_value {
            main_parameters.push(format!("{name}: {}", parameter.host.descend()));
            main_arguments.push(name.clone());
            continue;
        }

        let ownership = ownership_spelling(parameter);
        borrows = true;
        main_parameters.push(format!(
            "{name}: &'a {ownership} cpu.mem {}",
            parameter.host.descend()
        ));

        // The copy reads the host data, so it borrows it shared whatever ownership the
        // kernel declared -- a `uniq` can always be used as a `shrd`. The argument the
        // launch passes keeps the kernel's own ownership, below.
        main_bindings.push(format!(
            "let mut {} = gpu_alloc_copy(&uniq gpu, &shrd *{name});",
            device_copy_name(name)
        ));
        main_arguments.push(format!("&{ownership} {}", device_copy_name(name)));
    }

    let mut launch_dimensions = format!(
        "{}, {}",
        format_dimension(&kernel.dimensions.grid).map_err(|reason| {
            MacroError::at(
                format!(
                    "kernel `{}`: its grid is not a launchable size: {reason}",
                    kernel.function_name
                ),
                kernel.name_span,
            )
        })?,
        format_dimension(&kernel.dimensions.block).map_err(|reason| {
            MacroError::at(
                format!(
                    "kernel `{}`: its block is not a launchable size: {reason}",
                    kernel.function_name
                ),
                kernel.name_span,
            )
        })?
    );
    if !shared_memory_configurations.is_empty() {
        launch_dimensions.push_str("; ");
        launch_dimensions.push_str(&shared_memory_configurations.join(", "));
    }

    // `'a` is the lifetime of the host buffers the parameters borrow. A kernel taking
    // only values borrows nothing, and declaring a lifetime nothing uses is noise in
    // every diagnostic that quotes the signature.
    let lifetime = if borrows { " <'a>" } else { "" };

    let kernel_name = &kernel.function_name;
    let main_parameters = main_parameters.join(", ");
    let main_bindings = main_bindings.join("\n    ");
    let main_arguments = main_arguments.join(", ");

    Ok(format!(
        "
fn main({main_parameters}) -[t: cpu.thread]-> (){lifetime} {{
    let mut gpu = gpu_device(0);
    {main_bindings}
    {kernel_name}::<<<{launch_dimensions}>>>({main_arguments});
    ()
}}
",
    ))
}

fn ownership_spelling(parameter: &KernelParameter) -> &'static str {
    if parameter.is_uniq {
        "uniq"
    } else {
        "shrd"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_ty::HostTy;
    use descend::ast::{Dim, Dim1d, Nat};
    use proc_macro2::TokenStream;
    use std::str::FromStr;

    fn synthesized(source_text: &str, kernel_name: &str) -> Result<String, MacroError> {
        let source = crate::source::DescendSource::from_token_stream(
            TokenStream::from_str(source_text).expect("must tokenize"),
        );
        let code = descend::parser::SourceCode::new(source.text.clone());
        let unit = crate::source::parse_descend_source(&code)
            .unwrap_or_else(|error| panic!("fixture did not parse: {error}\n{}", source.text));
        let kernel = crate::kernel::Kernel::extract(&unit, kernel_name, &source)
            .unwrap_or_else(|error| panic!("fixture did not extract: {error}"));
        synthesize_main_function(&kernel)
    }

    /// A one-dimensional kernel with the parameter types given, so a test can name the
    /// shapes it cares about without writing Descend source for each.
    fn kernel_with(parameters: Vec<KernelParameter>) -> Kernel {
        Kernel {
            function_name: "gpu_k".to_string(),
            name_span: None,
            dimensions: crate::kernel::ExecutionDimensions {
                grid: Dim::X(Box::new(Dim1d(Nat::Lit(1)))),
                block: Dim::X(Box::new(Dim1d(Nat::Lit(64)))),
            },
            parameters,
        }
    }

    fn buffer(name: &str, is_uniq: bool) -> KernelParameter {
        KernelParameter {
            name: name.to_string(),
            span: None,
            is_uniq,
            is_shared_memory: false,
            is_by_value: false,
            host: HostTy::Array(
                Box::new(HostTy::Scalar(crate::host_ty::ScalarKind::I32)),
                crate::host_ty::ArrayLen::new(64, "64"),
            ),
        }
    }

    fn value(name: &str) -> KernelParameter {
        KernelParameter {
            name: name.to_string(),
            span: None,
            is_uniq: false,
            is_shared_memory: false,
            is_by_value: true,
            host: HostTy::Scalar(crate::host_ty::ScalarKind::I32),
        }
    }

    fn shared(name: &str) -> KernelParameter {
        KernelParameter {
            is_shared_memory: true,
            ..buffer(name, true)
        }
    }

    #[test]
    fn the_parameters_keep_the_names_the_kernel_gave_them() {
        let main = synthesized(
            "fn gpu_k<r1: prv, r2: prv>(
                 a_array: &r1 shrd gpu.global [i32; 64],
                 b_array: &r2 uniq gpu.global [i32; 64]
             ) -[grid: gpu.grid<X<1>, X<64>>]-> () {
                 sched block in grid { sched thread in block { () } }
             }",
            "gpu_k",
        )
        .expect("must synthesize");

        assert!(main.contains("a_array: &'a shrd cpu.mem"), "{main}");
        assert!(main.contains("b_array: &'a uniq cpu.mem"), "{main}");
        assert!(
            !main.contains("argument0"),
            "the numbered names should be gone: {main}"
        );
    }

    /// The reason the names matter: a type error in the synthesized `main` is the error
    /// the reader is shown, and it quotes this signature.
    #[test]
    fn a_diagnostic_would_name_what_the_reader_wrote() {
        let main = synthesized(
            "fn gpu_k<r: prv>(out_vec: &r uniq gpu.global [i32; 64])
                 -[grid: gpu.grid<X<1>, X<64>>]-> () {
                 sched block in grid { sched thread in block { () } }
             }",
            "gpu_k",
        )
        .expect("must synthesize");

        assert!(main.contains("out_vec"), "{main}");
        assert!(main.contains("&uniq out_vec_gpu"), "{main}");
    }

    #[test]
    fn a_by_value_parameter_is_passed_straight_through() {
        let main =
            synthesize_main_function(&kernel_with(vec![buffer("a_array", true), value("alpha")]))
                .expect("must synthesize");

        assert!(main.contains("alpha: i32"), "{main}");
        assert!(!main.contains("alpha_gpu"), "a value is not copied: {main}");
        assert!(main.contains("(&uniq a_array_gpu, alpha)"), "{main}");
    }

    #[test]
    fn a_shared_parameter_configures_the_launch_and_is_not_an_argument() {
        let main =
            synthesize_main_function(&kernel_with(vec![buffer("vec", false), shared("tmp")]))
                .expect("must synthesize");

        assert!(
            main.contains("X<64>; [i32; 64]"),
            "the shared parameter sizes the launch: {main}"
        );
        assert!(
            !main.contains("tmp:"),
            "the compiler removes shared parameters from the kernel signature: {main}"
        );
    }

    #[test]
    fn a_kernel_that_borrows_nothing_declares_no_lifetime() {
        let main =
            synthesize_main_function(&kernel_with(vec![value("alpha")])).expect("must synthesize");

        assert!(!main.contains("'a"), "nothing borrows: {main}");
        assert!(
            main.contains("fn main(alpha: i32) -[t: cpu.thread]-> () {"),
            "{main}"
        );
    }

    #[test]
    fn a_kernel_that_borrows_declares_the_lifetime_its_buffers_need() {
        let main = synthesize_main_function(&kernel_with(vec![buffer("a_array", true)]))
            .expect("must synthesize");

        assert!(main.contains("<'a>"), "{main}");
        assert!(main.contains("&'a uniq cpu.mem"), "{main}");
    }

    /// A parameter name Descend reserves would make the synthesized `main` unparsable,
    /// which is worse than an unreadable diagnostic.
    #[test]
    fn a_parameter_named_after_a_descend_keyword_falls_back_to_numbering() {
        let main = synthesize_main_function(&kernel_with(vec![
            buffer("shrd", true),
            buffer("other", false),
        ]))
        .expect("must synthesize");

        assert!(main.contains("argument0:"), "{main}");
        assert!(main.contains("argument1:"), "{main}");
    }

    #[test]
    fn a_parameter_that_would_shadow_the_device_local_falls_back_to_numbering() {
        let main = synthesize_main_function(&kernel_with(vec![buffer("gpu", true)]))
            .expect("must synthesize");

        assert!(main.contains("argument0:"), "{main}");
        assert!(!main.contains("gpu:"), "{main}");
    }

    /// `x` and `x_gpu` both want a local called `x_gpu`.
    #[test]
    fn two_parameters_contending_for_one_device_local_fall_back_to_numbering() {
        let main =
            synthesize_main_function(&kernel_with(vec![buffer("x", true), buffer("x_gpu", true)]))
                .expect("must synthesize");

        assert!(
            main.contains("argument0:") && main.contains("argument1:"),
            "{main}"
        );
    }

    #[test]
    fn a_usable_name_is_not_disturbed_by_a_near_miss() {
        // `gpu_` is not `gpu`, and `x_gp` does not produce `x_gpu`.
        let main = synthesize_main_function(&kernel_with(vec![
            buffer("gpu_", true),
            buffer("x_gp", false),
        ]))
        .expect("must synthesize");

        assert!(main.contains("gpu_: &'a uniq cpu.mem"), "{main}");
        assert!(main.contains("x_gp: &'a shrd cpu.mem"), "{main}");
    }

    /// `DESCEND_KEYWORDS` is a copy of a private peg rule, so nothing stops the two
    /// drifting. This reads the rule back out of the parser and compares, which turns
    /// an assumption into a checked invariant.
    #[test]
    fn the_keyword_list_matches_the_parser_rule() {
        let parser_source =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../src/parser/mod.rs"))
                .expect("the parser source must be readable");

        let mut spelled_in_the_rule = quoted_alternatives(&parser_source, "rule keyword()");

        // `let`("prov")? is the rule's one optional continuation, and the only place
        // where what it spells and what it accepts come apart: `prov` is a quoted string
        // that is not a keyword on its own, and `letprov` is a keyword the rule never
        // spells as a single string.
        spelled_in_the_rule.retain(|keyword| keyword != "prov");
        spelled_in_the_rule.push("letprov".to_string());

        // `view_name()` is included by reference, so its alternatives are keywords too
        // even though they are spelled in a rule of their own.
        spelled_in_the_rule.extend(quoted_alternatives(&parser_source, "rule view_name()"));

        let mut expected = spelled_in_the_rule;
        expected.sort();
        expected.dedup();

        let mut actual: Vec<String> = DESCEND_KEYWORDS
            .iter()
            .map(|keyword| (*keyword).to_string())
            .collect();
        actual.sort();
        actual.dedup();

        assert_eq!(
            actual, expected,
            "DESCEND_KEYWORDS has drifted from the parser's rules"
        );
    }

    /// The quoted strings a peg rule offers as alternatives.
    ///
    /// The right-hand side is found by matching parentheses rather than by looking for
    /// the first `)`, because `keyword()` nests one: `"let"("prov")?` would otherwise
    /// end the scan before `if`, `sched` and every later alternative.
    fn quoted_alternatives(parser_source: &str, rule_header: &str) -> Vec<String> {
        let rule_start = parser_source
            .find(rule_header)
            .unwrap_or_else(|| panic!("the `{rule_header}` rule must still exist"));
        let from_the_header = &parser_source[rule_start..];

        // `= (` opens the alternatives. The header's own parentheses and the `-> ()`
        // return type both come before it, and neither spells `= (`.
        let opening = from_the_header
            .find("= (")
            .unwrap_or_else(|| panic!("`{rule_header}` must still assign its alternatives"));
        let alternatives = &from_the_header[opening + "= ".len()..];

        let mut depth = 0usize;
        let mut end_of_alternatives = None;
        for (offset, character) in alternatives.char_indices() {
            match character {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end_of_alternatives = Some(offset);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end_of_alternatives = end_of_alternatives
            .unwrap_or_else(|| panic!("`{rule_header}` must still close its alternatives"));

        alternatives[..end_of_alternatives]
            .split('"')
            .skip(1)
            .step_by(2)
            .filter(|keyword| !keyword.is_empty())
            .map(|keyword| keyword.to_string())
            .collect()
    }
}
