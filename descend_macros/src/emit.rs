use crate::artifact::{LaunchConfig, Ptx};
use crate::diag::MacroError;
use crate::kernel::Kernel;
use crate::source::DescendSource;
use proc_macro2::{Delimiter, Group, Ident, Literal, TokenStream, TokenTree};
use quote::{format_ident, quote};
use std::str::FromStr;

pub fn generate_launch_function(
    kernel: &Kernel,
    launch: &LaunchConfig,
    compiled: &Ptx,
) -> Result<TokenStream, MacroError> {
    let kernel_name = &kernel.function_name;

    let mut parameters: Vec<TokenStream> = vec![quote! { stream: &::descend_host::Stream }];
    let mut argument_pushes: Vec<TokenStream> = Vec::new();
    for parameter in kernel
        .parameters
        .iter()
        .filter(|parameter| !parameter.is_shared_memory)
    {
        let rust_name = rust_parameter_name(&parameter.name, kernel_name)
            .map_err(|message| MacroError::at(message, parameter.span))?;

        if parameter.is_by_value {
            let parameter_ident = format_ident!("{}", rust_name);
            let value_type = TokenStream::from_str(&parameter.host.rust()).map_err(|error| {
                MacroError::at(
                    format!(
                        "kernel `{kernel_name}`: parameter `{}` has a type the generator cannot \
                             spell in Rust: {error}",
                        parameter.name
                    ),
                    parameter.span,
                )
            })?;
            parameters.push(quote! { #parameter_ident: #value_type });
            argument_pushes.push(quote! { kernel.push_arg_val(&#parameter_ident)?; });
            continue;
        }

        let element_count = parameter.host.total_elements();

        let parameter_ident = format_ident!("{}", rust_name);
        let element = format_ident!("{}", parameter.host.base().rust());
        let element_count_literal = Literal::usize_unsuffixed(element_count);
        let buffer_type = quote! { ::descend_host::GpuBuffer<[#element; #element_count_literal]> };

        if parameter.is_uniq {
            parameters.push(quote! { #parameter_ident: &mut #buffer_type });
            argument_pushes.push(quote! { kernel.push_arg_uniq(#parameter_ident)?; });
        } else {
            parameters.push(quote! { #parameter_ident: &#buffer_type });
            argument_pushes.push(quote! { kernel.push_arg_shrd(#parameter_ident)?; });
        }
    }

    let kernel_ident = format_ident!("{}", kernel_name);
    let entry = &compiled.entry;
    let ptx_source = &compiled.text;

    let (grid_x, grid_y, grid_z) = (launch.grid.x, launch.grid.y, launch.grid.z);
    let (block_x, block_y, block_z) = (launch.block.x, launch.block.y, launch.block.z);
    let shared_memory_bytes = Literal::u32_unsuffixed(launch.shared_memory_bytes.value());

    Ok(quote! {
        fn #kernel_ident(
            #(#parameters),*
        ) -> ::core::result::Result<(), ::descend_host::error::HostError> {
            const PTX_SOURCE: &str = #ptx_source;
            let mut kernel = ::descend_host::Kernel::new(
                #entry,
                ::descend_host::KernelLaunchConfig {
                    grid_dim: (#grid_x, #grid_y, #grid_z),
                    block_dim: (#block_x, #block_y, #block_z),
                    shared_mem_bytes: #shared_memory_bytes,
                },
            );
            #(#argument_pushes)*
            let launcher = ::descend_host::KernelLauncher::new(stream, PTX_SOURCE)?;
            let launch_result = unsafe { launcher.launch(kernel) };
            stream.synchronize()?;
            launch_result
        }
    })
}

fn rust_parameter_name(name: &str, kernel_name: &str) -> Result<String, String> {
    const UNRAWABLE: &[&str] = &["self", "Self", "crate", "super", "_"];
    const KEYWORDS: &[&str] = &[
        "as", "async", "await", "break", "const", "continue", "dyn", "else", "enum", "extern",
        "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut",
        "pub", "ref", "return", "static", "struct", "trait", "true", "type", "union", "unsafe",
        "use", "where", "while", "abstract", "become", "box", "do", "final", "macro", "override",
        "priv", "try", "typeof", "unsized", "virtual", "yield", "gen",
    ];

    if UNRAWABLE.contains(&name) {
        return Err(format!(
            "kernel `{kernel_name}`: parameter `{name}` cannot name a Rust function parameter"
        ));
    }
    if KEYWORDS.contains(&name) {
        return Ok(format!("r#{name}"));
    }
    Ok(name.to_string())
}

pub fn format_rust_struct(
    struct_decl: &descend::ast::StructDecl,
    source: &DescendSource,
) -> Result<String, MacroError> {
    let struct_name = &struct_decl.ident.name;
    let struct_span = crate::kernel::span_of(&struct_decl.ident, source);

    let mut output = String::new();
    output.push_str("#[repr(C)]\n#[derive(Clone, Copy, Debug, PartialEq)]\npub struct ");
    output.push_str(struct_name);
    output.push_str(" {\n");
    for (field_ident, field_dty) in &struct_decl.fields {
        let field_name = &field_ident.name;
        output.push_str("    pub ");
        output.push_str(field_name);
        output.push_str(": ");
        output.push_str(
            &crate::host_ty::from_dty(field_dty)
                .map_err(|reason| {
                    MacroError::at(
                        format!("struct `{struct_name}`: field `{field_name}` {reason}"),
                        crate::kernel::span_of(field_ident, source).or(struct_span),
                    )
                })?
                .rust(),
        );
        output.push_str(",\n");
    }
    output.push_str("}\n");
    output.push_str(&format!(
        "unsafe impl ::descend_host::traits::DescendType for {struct_name} {{}}\n",
    ));
    Ok(output)
}

fn carrier_name(kernel_name: &str) -> Ident {
    format_ident!("__descend_kernel_{}", kernel_name)
}

pub fn carrier_macro(kernel_name: &str, source_token_stream: &TokenStream) -> TokenStream {
    let carrier = carrier_name(kernel_name);

    quote! {
        #[allow(unused_macros)]
        macro_rules! #carrier {
            (@fn $name:ident, $($args:tt)*) => {
                ::descend_macros::__descend_fn_internal! {
                    { #source_token_stream }, $name, $($args)*
                }
            };
        }
    }
}

pub fn carrier_invocation(
    kernel: &Ident,
    bindings: TokenStream,
) -> Result<TokenStream, MacroError> {
    let mut carrier_args = TokenStream::from_str(&format!("@fn {kernel},"))
        .map_err(|error| format!("malformed instantiation: {error}"))?;
    carrier_args.extend(bindings);

    let mut invocation = format!("{}!", carrier_name(&kernel.to_string()))
        .parse::<TokenStream>()
        .expect("the carrier's name is an identifier");
    invocation.extend([TokenTree::Group(Group::new(Delimiter::Brace, carrier_args))]);
    Ok(invocation)
}

pub fn block_yielding(items: TokenStream, name: &Ident) -> TokenStream {
    let mut block = items;
    block.extend([TokenTree::Ident(name.clone())]);
    TokenStream::from(TokenTree::Group(Group::new(Delimiter::Brace, block)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declared(source_text: &str) -> Result<String, MacroError> {
        use proc_macro2::TokenStream;
        use std::str::FromStr;

        let source = crate::source::DescendSource::from_token_stream(
            TokenStream::from_str(source_text).expect("must tokenize"),
        );
        let code = descend::parser::SourceCode::new(source.text.clone());
        let unit = crate::capture::with_captured_output(|| descend::parser::parse(&code))
            .0
            .unwrap_or_else(|_| panic!("fixture did not parse:\n{}", source.text));

        let struct_decl = unit
            .items
            .iter()
            .find_map(|item| match item {
                descend::ast::Item::StructDecl(struct_decl) => Some(struct_decl),
                _ => None,
            })
            .expect("fixture declares a struct");

        format_rust_struct(struct_decl, &source)
    }

    #[test]
    fn every_scalar_width_maps_to_its_rust_counterpart() {
        let rust =
            declared("struct Scalars { a: i32, b: u8, c: u32, d: u64, e: f32, f: f64, g: bool }")
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
        let rust =
            declared("struct Outer { inner: Inner, id: u32 } struct Inner { x: i32 }").unwrap();
        assert!(rust.contains("pub inner: Inner"), "{rust}");
    }

    #[test]
    fn an_unsupported_field_type_is_refused_with_the_generic_sentence() {
        let error = declared("struct Particle { pos: (f32, f32), id: i32 }").unwrap_err();
        assert!(error.message.contains("field `pos`"), "{error}");
        assert!(
            error.message.contains(crate::host_ty::UNSUPPORTED_TYPE),
            "{error}"
        );
    }

    #[test]
    fn a_rejected_field_is_pointed_at_rather_than_the_block_it_is_in() {
        let error = declared("struct Particle { pos: (f32, f32), id: i32 }").unwrap_err();
        assert!(
            error.span.is_some(),
            "the refusal should point at `pos`: {error}"
        );
    }

    #[test]
    fn a_unit_field_is_rejected() {
        let error = declared("struct Empty { nothing: () }").unwrap_err();
        assert!(error.message.contains("field `nothing`"), "{error}");
    }

    #[test]
    fn an_unresolved_size_in_a_field_is_reported() {
        let error = declared("struct Open { cells: [i32; N] }").unwrap_err();
        assert!(error.message.contains("unresolved size"), "{error}");
        assert!(error.message.contains("field `cells`"), "{error}");
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
