use crate::artifact::{LaunchConfig, Ptx};
use crate::diag::MacroError;
use crate::invocation::DimensionAnnotation;
use crate::kernel::{Kernel, ParameterType};
use descend::ast::{Memory, Ownership};
use proc_macro2::{Delimiter, Group, Ident, Literal, Punct, Spacing, TokenStream, TokenTree};
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
    for parameter in &kernel.parameters {
        if matches!(
            &parameter.kind,
            ParameterType::Reference {
                memory: Memory::GpuShared,
                ..
            }
        ) {
            continue;
        }

        let rust_name = rust_parameter_name(&parameter.name, kernel_name)
            .map_err(|message| MacroError::at(message, parameter.span))?;
        let parameter_ident = format_ident!("{}", rust_name);

        match &parameter.kind {
            ParameterType::Value(_) => {
                let value_type =
                    TokenStream::from_str(&parameter.host.rust()).map_err(|error| {
                        MacroError::at(
                            format!(
                                "kernel `{kernel_name}`: parameter `{}` has a type the generator \
                         cannot spell in Rust: {error}",
                                parameter.name
                            ),
                            parameter.span,
                        )
                    })?;
                parameters.push(quote! { #parameter_ident: #value_type });
                argument_pushes.push(quote! { kernel.push_arg_val(&#parameter_ident)?; });
            }
            ParameterType::Reference { ownership, .. } => {
                let element_count = parameter.host.total_elements();
                let element = format_ident!("{}", parameter.host.base().rust());
                let element_count_literal = Literal::usize_unsuffixed(element_count);
                let buffer_type =
                    quote! { ::descend_host::GpuBuffer<[#element; #element_count_literal]> };

                match ownership {
                    Ownership::Uniq => {
                        parameters.push(quote! { #parameter_ident: &mut #buffer_type });
                        argument_pushes.push(quote! { kernel.push_arg_uniq(#parameter_ident)?; });
                    }
                    Ownership::Shrd => {
                        parameters.push(quote! { #parameter_ident: &#buffer_type });
                        argument_pushes.push(quote! { kernel.push_arg_shrd(#parameter_ident)?; });
                    }
                }
            }
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

    const GENERATED_BINDINGS: &[&str] = &[
        "stream",
        "kernel",
        "launcher",
        "launch_result",
        "PTX_SOURCE",
    ];

    if UNRAWABLE.contains(&name) {
        return Err(format!(
            "kernel `{kernel_name}`: parameter `{name}` cannot name a Rust function parameter"
        ));
    }
    if GENERATED_BINDINGS.contains(&name) {
        return Err(format!(
            "kernel `{kernel_name}`: parameter `{name}` collides with a binding the generated \
             launch function makes itself, so rename the parameter"
        ));
    }
    if KEYWORDS.contains(&name) {
        return Ok(format!("r#{name}"));
    }
    Ok(name.to_string())
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
    dimension_annotation: Option<&DimensionAnnotation>,
    bindings: TokenStream,
) -> Result<TokenStream, MacroError> {
    let mut carrier_args = TokenStream::from_str("@fn ")
        .map_err(|error| format!("malformed instantiation: {error}"))?;
    carrier_args.extend([TokenTree::Ident(kernel.clone())]);
    carrier_args.extend([TokenTree::Punct(Punct::new(',', Spacing::Alone))]);
    if let Some(dimension_annotation) = dimension_annotation {
        carrier_args.extend([dimension_annotation.to_token_tree()]);
        carrier_args.extend([TokenTree::Punct(Punct::new(',', Spacing::Alone))]);
    }
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

    #[test]
    fn a_parameter_named_after_a_binding_the_generator_makes_is_refused() {
        for name in [
            "stream",
            "kernel",
            "launcher",
            "launch_result",
            "PTX_SOURCE",
        ] {
            let error = rust_parameter_name(name, "gpu_k").unwrap_err();
            assert!(
                error.contains(name) && error.contains("gpu_k"),
                "unhelpful error for `{name}`: {error}"
            );
            assert!(error.contains("collides"), "{error}");
        }
    }
}
