use crate::compiler::CompiledKernel;
use crate::diag::MacroError;
use crate::kernel::Kernel;
use crate::nat::eval_dim;
use crate::shared_mem::check_shared_mem;
use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};
use std::str::FromStr;

pub fn generate_launch_fn(
    kernel: &Kernel,
    compiled: &CompiledKernel,
    item_name: &str,
) -> Result<TokenStream, MacroError> {
    let function = &kernel.function_name;

    check_shared_mem(kernel, compiled.shared_mem_bytes)?;

    let (Some(grid), Some(block)) = (&kernel.grid_dim, &kernel.block_dim) else {
        return Err(MacroError::at(
            format!("kernel `{function}` must declare explicit grid and block dimensions"),
            kernel.name_span,
        ));
    };
    let grid_dim = eval_dim(grid)?;
    let block_dim = eval_dim(block)?;

    let mut parameters: Vec<TokenStream> = vec![quote! { stream: &::descend_host::Stream }];
    let mut pushes: Vec<TokenStream> = Vec::new();
    for param in kernel.params.iter().filter(|p| !p.is_shared_mem) {
        let binding = rust_parameter_name(&param.name, function)
            .map_err(|message| MacroError::at(message, param.span))?;

        if param.is_by_value {
            let name = format_ident!("{}", binding);
            let value_ty = TokenStream::from_str(&param.host.rust()).map_err(|error| {
                MacroError::at(
                    format!(
                        "kernel `{function}`: parameter `{}` has a type the generator cannot \
                             spell in Rust: {error}",
                        param.name
                    ),
                    param.span,
                )
            })?;
            parameters.push(quote! { #name: #value_ty });
            pushes.push(quote! { kernel.push_arg_val(&#name)?; });
            continue;
        }

        let elements = param.host.elements().ok_or_else(|| {
            MacroError::at(
                format!(
                    "kernel `{function}`: parameter `{}` has more elements than can be counted",
                    param.name
                ),
                param.span,
            )
        })?;

        let name = format_ident!("{}", binding);
        let elem = format_ident!("{}", param.host.base().rust());
        let elements = Literal::usize_unsuffixed(elements as usize);
        let contents = quote! { ::descend_host::GpuBuffer<[#elem; #elements]> };

        if param.is_uniq {
            parameters.push(quote! { #name: &mut #contents });
            pushes.push(quote! { kernel.push_arg_uniq(#name)?; });
        } else {
            parameters.push(quote! { #name: &#contents });
            pushes.push(quote! { kernel.push_arg_shrd(#name)?; });
        }
    }

    let function_name = format_ident!("{}", item_name);
    let entry = &compiled.mangled_name;
    let ptx = Literal::byte_string(compiled.ptx.as_bytes());

    let (grid_x, grid_y, grid_z) = grid_dim;
    let (block_x, block_y, block_z) = block_dim;
    let shared_mem_bytes =
        Literal::u32_unsuffixed(u32::try_from(compiled.shared_mem_bytes).map_err(|_| {
            MacroError::at(
                format!(
                "kernel `{function}` requests {} bytes of shared memory, more than a launch can \
                     express",
                    compiled.shared_mem_bytes
                ),
                kernel.name_span,
            )
        })?);

    Ok(quote! {
        #[allow(dead_code)]
        pub fn #function_name(
            #(#parameters),*
        ) -> ::core::result::Result<(), ::descend_host::error::HostError> {
            const PTX_BYTES: &[u8] = #ptx;
            let mut kernel = ::descend_host::Kernel::new(
                #entry,
                ::descend_host::KernelLaunchConfig {
                    grid_dim: (#grid_x, #grid_y, #grid_z),
                    block_dim: (#block_x, #block_y, #block_z),
                    shared_mem_bytes: #shared_mem_bytes,
                },
            );
            #(#pushes)*
            let launcher = ::descend_host::KernelLauncher::new(
                stream,
                ::core::str::from_utf8(PTX_BYTES).unwrap(),
            )?;
            let launch_result = unsafe { launcher.launch(kernel) };
            stream.synchronize()?;
            launch_result
        }
    })
}

fn rust_parameter_name(name: &str, function: &str) -> Result<String, String> {
    const UNRAWABLE: [&str; 5] = ["self", "Self", "crate", "super", "_"];
    const KEYWORDS: [&str; 49] = [
        "as", "async", "await", "break", "const", "continue", "dyn", "else", "enum", "extern",
        "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut",
        "pub", "ref", "return", "static", "struct", "trait", "true", "type", "union", "unsafe",
        "use", "where", "while", "abstract", "become", "box", "do", "final", "macro", "override",
        "priv", "try", "typeof", "unsized", "virtual", "yield", "gen",
    ];

    if UNRAWABLE.contains(&name) {
        return Err(format!(
            "kernel `{function}`: parameter `{name}` cannot name a Rust function parameter"
        ));
    }
    if KEYWORDS.contains(&name) {
        return Ok(format!("r#{name}"));
    }
    Ok(name.to_string())
}

/// The Rust definition of a type a `descend!` block declares.
///
/// `#[repr(C)]` so the field order and offsets are specified rather than left to the
/// compiler, and `DescendType` so the type can fill a `GpuBuffer`: the macro derived
/// this definition from the declaration, so it is the only thing that can vouch for it.
///
/// Note that a kernel taking a declared type does not compile yet — Descend's code
/// generator refers to a struct by name but never emits its definition, so nvcc fails
/// with `identifier "Pair" is undefined`. Modelling the type here is what makes that a
/// one-sided gap: once the generator emits the C++ definition, the host side is
/// already in place.
pub fn format_rust_struct(s: &descend::ast::StructDecl) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("#[repr(C)]\n#[derive(Clone, Copy, Debug, PartialEq)]\npub struct ");
    out.push_str(&s.ident.name);
    out.push_str(" {\n");
    for (ident, dty) in &s.fields {
        out.push_str("    pub ");
        out.push_str(&ident.name);
        out.push_str(": ");
        out.push_str(
            &crate::host_ty::host_ty(dty)
                .map_err(|reason| {
                    format!(
                        "struct `{}`: field `{}` {}",
                        s.ident.name,
                        ident.name,
                        reason.reason()
                    )
                })?
                .rust(),
        );
        out.push_str(",\n");
    }
    out.push_str("}\n");
    out.push_str(&format!(
        "unsafe impl ::descend_host::traits::DescendType for {} {{}}\n",
        s.ident.name
    ));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Rust definition `descend!` emits for a type it declares.
    fn declared(source: &str) -> Result<String, String> {
        use proc_macro2::TokenStream;
        use std::str::FromStr;

        let flattened =
            crate::parsing::flatten_source(TokenStream::from_str(source).expect("must tokenize"));
        let code = descend::parser::SourceCode::new(flattened.text.clone());
        let unit = crate::capture::with_captured_output(|| descend::parser::parse(&code))
            .0
            .unwrap_or_else(|_| panic!("fixture did not parse:\n{}", flattened.text));

        let declared = unit
            .items
            .iter()
            .find_map(|item| match item {
                descend::ast::Item::StructDecl(s) => Some(s),
                _ => None,
            })
            .expect("fixture declares a struct");

        format_rust_struct(declared)
    }

    #[test]
    fn every_scalar_width_maps_to_its_rust_counterpart() {
        let rust = declared(
            "struct Scalars { a: i32, b: u8, c: u32, d: u64, e: f32, f: f64, g: bool }",
        )
        .unwrap();
        for field in [
            "pub a: i32",
            "pub b: u8",
            "pub c: u32",
            "pub d: u64",
            "pub e: f32",
            "pub f: f64",
            "pub g: bool",
        ] {
            assert!(rust.contains(field), "{field} missing from:\n{rust}");
        }
    }

    #[test]
    fn the_declared_type_is_repr_c_and_transferable() {
        let rust = declared("struct Pair { a: i32, b: i32 }").unwrap();
        assert!(rust.contains("#[repr(C)]"), "{rust}");
        assert!(
            rust.contains("unsafe impl ::descend_host::traits::DescendType for Pair"),
            "a declared type must be usable in a GpuBuffer:\n{rust}"
        );
    }

    #[test]
    fn an_array_field_keeps_its_length() {
        let rust = declared("struct Window { history: [i32; 4] }").unwrap();
        assert!(rust.contains("pub history: [i32; 4]"), "{rust}");
    }

    #[test]
    fn a_nested_array_field_keeps_both_lengths() {
        let rust = declared("struct Tile { rows: [[f32; 2]; 3] }").unwrap();
        assert!(rust.contains("pub rows: [[f32; 2]; 3]"), "{rust}");
    }

    #[test]
    fn an_array_length_may_be_an_expression() {
        let rust = declared("struct Block { cells: [i32; 4*8] }").unwrap();
        assert!(rust.contains("pub cells: [i32; 32]"), "{rust}");
    }

    #[test]
    fn a_nested_declared_type_is_referred_to_by_name() {
        // `declared` reads the first struct, so the nesting one comes first.
        let rust =
            declared("struct Outer { inner: Inner, id: u32 } struct Inner { x: i32 }").unwrap();
        assert!(rust.contains("pub inner: Inner"), "{rust}");
    }

    #[test]
    fn a_tuple_field_is_rejected_rather_than_silently_narrowed() {
        let error = declared("struct Particle { pos: (f32, f32), id: i32 }").unwrap_err();
        assert!(error.contains("field `pos`"), "{error}");
        assert!(error.contains("tuple"), "{error}");
        assert!(
            error.contains("memory layout"),
            "the reason must be the layout, not just a refusal: {error}"
        );
    }

    #[test]
    fn a_unit_field_is_rejected() {
        let error = declared("struct Empty { nothing: () }").unwrap_err();
        assert!(error.contains("field `nothing`"), "{error}");
    }

    #[test]
    fn an_unresolved_size_in_a_field_is_reported() {
        let error = declared("struct Open { cells: [i32; N] }").unwrap_err();
        assert!(error.contains("unresolved size"), "{error}");
        assert!(error.contains("field `cells`"), "{error}");
    }

    #[test]
    fn an_ordinary_parameter_name_is_kept() {
        assert_eq!(rust_parameter_name("a_array", "k").unwrap(), "a_array");
    }

    #[test]
    fn a_rust_keyword_becomes_a_raw_identifier() {
        assert_eq!(rust_parameter_name("type", "k").unwrap(), "r#type");
        assert_eq!(rust_parameter_name("move", "k").unwrap(), "r#move");
        assert_eq!(rust_parameter_name("ref", "k").unwrap(), "r#ref");
    }

    #[test]
    fn an_identifier_rust_cannot_make_raw_is_reported() {
        for name in ["self", "Self", "crate", "super", "_"] {
            let error = rust_parameter_name(name, "gpu_k").unwrap_err();
            assert!(
                error.contains(name) && error.contains("gpu_k"),
                "unhelpful error for `{name}`: {error}"
            );
        }
    }
}
