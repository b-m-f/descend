#![allow(dead_code)]
#![allow(unreachable_patterns)]
use descend::ast::*;
use proc_macro::{TokenStream, TokenTree};
use std::collections::HashMap;

pub struct LaunchConfig {
    pub stream_expr: proc_macro::TokenStream,
    pub generic_to_size: HashMap<String, String>,
    pub array_args: Vec<proc_macro::TokenStream>,
}

impl LaunchConfig {
    /// Parses the launch configuration block from the `descend_launch!` internal macro invocation.
    ///
    /// What it does: Extracts the generic bindings (like `<N: 256>`) and runtime arguments (like `&mut a_gpu`) into a `LaunchConfig` struct.
    /// Why it is needed: Necessary to resolve abstract macro sizes to concrete integer bounds required for NVCC compilation and to capture the host runtime parameters to embed back into the CUDA call.
    pub fn parse(input: TokenStream) -> Result<Self, String> {
        let mut tokens = input.into_iter().peekable();

        // Parse stream_expr until the first comma
        let mut stream_tokens = Vec::new();
        while let Some(tt) = tokens.peek() {
            if let proc_macro::TokenTree::Punct(p) = tt {
                if p.as_char() == ',' {
                    break;
                }
            }
            stream_tokens.push(tokens.next().unwrap());
        }

        if let Some(proc_macro::TokenTree::Punct(p)) = tokens.peek() {
            if p.as_char() == ',' {
                tokens.next(); // consume the comma
            }
        }

        let mut stream_expr = proc_macro::TokenStream::new();
        stream_expr.extend(stream_tokens);

        let mut generic_to_size = HashMap::new();

        // Check if there is `<` next
        if let Some(TokenTree::Punct(p)) = tokens.peek() {
            if p.as_char() == '<' {
                tokens.next(); // consume `<`

                // Read generic key-value pairs until `>`
                while let Some(tt) = tokens.peek() {
                    if let TokenTree::Punct(p) = tt {
                        if p.as_char() == '>' {
                            tokens.next(); // consume `>`
                            break;
                        }
                    }

                    let gen_ident = match tokens.next() {
                        Some(TokenTree::Ident(id)) => id,
                        _ => return Err("Expected generic identifier in launch config".to_string()),
                    };

                    match tokens.next() {
                        Some(TokenTree::Punct(p)) if p.as_char() == ':' => {}
                        _ => return Err("Expected ':' in generic argument".to_string()),
                    }

                    let val = match tokens.next() {
                        Some(TokenTree::Literal(lit)) => lit.to_string(),
                        _ => return Err("Expected integer literal for generic value".to_string()),
                    };

                    generic_to_size.insert(gen_ident.to_string(), val);

                    if let Some(TokenTree::Punct(p)) = tokens.peek() {
                        if p.as_char() == ',' {
                            tokens.next(); // consume `,`
                        }
                    }
                }
            }
        }

        // if there's a comma after generics, consume it
        if let Some(TokenTree::Punct(p)) = tokens.peek() {
            if p.as_char() == ',' {
                tokens.next();
            }
        }

        let mut args = Vec::new();
        while let Some(tt) = tokens.next() {
            if let TokenTree::Punct(p) = &tt {
                if p.as_char() == ',' {
                    continue;
                }
            }

            // Collect tokens until the next comma
            let mut arg_tokens = vec![tt];
            while let Some(next_tt) = tokens.peek() {
                if let TokenTree::Punct(p) = next_tt {
                    if p.as_char() == ',' {
                        break;
                    }
                }
                arg_tokens.push(tokens.next().unwrap());
            }

            let mut arg_ts = TokenStream::new();
            for t in arg_tokens {
                arg_ts.extend(std::iter::once(t));
            }
            args.push(arg_ts);
        }

        Ok(LaunchConfig {
            stream_expr,
            generic_to_size,
            array_args: args,
        })
    }
}

/// Wraps the invocation of the Descend language parser.
///
/// What it does: Calls the `descend::parser::parse` method to extract the Abstract Syntax Tree (AST).
/// Why it is needed: Safely converts the user's macro code into a semantic structure (`CompilUnit`) we can navigate to identify `Item::FunDef` nodes for code generation.
pub fn parse_descend_source<'a>(
    source: &'a descend::parser::SourceCode,
) -> Result<CompilUnit<'a>, String> {
    descend::parser::parse(source)
        .map_err(|e| format!("Failed to parse descend macro input: {:?}", e))
}

/// Translates Descend execution block dimensionality expressions into strings.
///
/// What it does: Flattens `Exec::GpuGrid` configurations into comma-separated text layouts.
/// Why it is needed: Converts the internal AST coordinate types to text strings capable of being evaluated for CUDA grid launch configurations.
pub fn parse_grid_dim(exec_kind: &ExecExprKind) -> Option<String> {
    if let BaseExec::GpuGrid(dim_x, dim_y) = &exec_kind.base {
        let x_str = format_dim(dim_x);
        let y_str = format_dim(dim_y);
        return Some(format!("{}, {}", x_str, y_str));
    }
    None
}

/// Converts 1D, 2D, and 3D native Descend dimensions into text.
///
/// What it does: Matches inner `Dim::X`, `Dim::XY`, etc. variants and delegates to `nat_to_string`.
/// Why it is needed: Allows extraction of complex algebraic structures mapping directly to CUDA thread layouts.
pub fn format_dim(dim: &descend::ast::Dim) -> String {
    match dim {
        descend::ast::Dim::X(nat) => format!("X<{}>", nat_to_string(&nat.0)),
        descend::ast::Dim::Y(nat) => format!("Y<{}>", nat_to_string(&nat.0)),
        descend::ast::Dim::Z(nat) => format!("Z<{}>", nat_to_string(&nat.0)),
        descend::ast::Dim::XY(dim2d) => format!(
            "XY<{}, {}>",
            nat_to_string(&dim2d.0),
            nat_to_string(&dim2d.1)
        ),
        descend::ast::Dim::XZ(dim2d) => format!(
            "XZ<{}, {}>",
            nat_to_string(&dim2d.0),
            nat_to_string(&dim2d.1)
        ),
        descend::ast::Dim::YZ(dim2d) => format!(
            "YZ<{}, {}>",
            nat_to_string(&dim2d.0),
            nat_to_string(&dim2d.1)
        ),
        descend::ast::Dim::XYZ(dim3d) => format!(
            "XYZ<{}, {}, {}>",
            nat_to_string(&dim3d.0),
            nat_to_string(&dim3d.1),
            nat_to_string(&dim3d.2)
        ),
    }
}

/// Unrolls mathematical expressions and abstract number representations back to text strings.
///
/// What it does: Recursively visits `Nat::Lit`, `Nat::BinOp`, etc., converting tree data structures into calculable strings.
/// Why it is needed: Maintains mathematical operators supplied in grid sizes (e.g. `n/bs`) intact so they can be processed and verified cleanly.
fn nat_to_string(nat: &Nat) -> String {
    match nat {
        Nat::Lit(v) => v.to_string(),
        Nat::Ident(id) => id.name.to_string(),
        Nat::BinOp(op, n1, n2) => {
            let op_str = match op {
                descend::ast::BinOpNat::Add => "+",
                descend::ast::BinOpNat::Sub => "-",
                descend::ast::BinOpNat::Mul => "*",
                descend::ast::BinOpNat::Div => "/",
                descend::ast::BinOpNat::Mod => "%",
            };
            format!("{} {} {}", nat_to_string(n1), op_str, nat_to_string(n2))
        }
        Nat::App(id, args) => {
            let args_str: Vec<String> = args.iter().map(nat_to_string).collect();
            format!("{}({})", id.name, args_str.join(", "))
        }
        _ => "?".to_string(),
    }
}

/// Scans nested type hierarchies looking for terminal array limits.
///
/// What it does: Follows inner data structures of vectors until locating the bounding `Nat` object.
/// Why it is needed: Required for computing buffer allocations statically at compile time.
pub fn extract_size_from_ty(ty: &Ty) -> String {
    if let TyKind::Data(dty) = &ty.ty {
        let mut current_dty = if let DataTyKind::Ref(ref_dty) = &dty.dty {
            &ref_dty.dty.dty
        } else {
            &dty.dty
        };

        let mut dims = Vec::new();
        while let DataTyKind::Array(elem_ty, nat) = current_dty {
            dims.push(nat_to_string(nat));
            current_dty = &elem_ty.dty;
        }

        if !dims.is_empty() {
            return dims.join(" * ");
        }
    }
    "1".to_string()
}

/// Maps abstract GPU global structures to host-level representation strings.
///
/// What it does: Emits Rust types representing mapped memory slices like `&shrd cpu.mem [i32; ...] `.
/// Why it is needed: Reconstitutes synthetically the CPU-side parameter mappings mirroring GPU resources.
pub fn format_cpu_ty(ty: &Ty) -> String {
    if let TyKind::Data(dty) = &ty.ty {
        if let DataTyKind::Ref(ref_dty) = &dty.dty {
            let own_str = match ref_dty.own {
                descend::ast::Ownership::Shrd => "shrd",
                descend::ast::Ownership::Uniq => "uniq",
            };
            let inner_ty_str = format_array_dty(&ref_dty.dty);
            return format!("&'a {} cpu.mem {}", own_str, inner_ty_str);
        }
    }
    "UNKNOWN_TY".to_string()
}

/// Extracts the dimensionality layout of a nested shared block array.
///
/// What it does: Traverses `GpuShared` AST type parameters to isolate their structure.
/// Why it is needed: Necessary to determine the amount of runtime dynamic memory to allocate.
pub fn format_shared_array_ty(ty: &Ty) -> String {
    if let TyKind::Data(dty) = &ty.ty {
        if let DataTyKind::Ref(ref_dty) = &dty.dty {
            return format_array_dty(&ref_dty.dty);
        }
    }
    "UNKNOWN_ARRAY".to_string()
}

/// Builds text approximations for AST arrays.
///
/// What it does: Maps `DataTyKind` variants recursively to scalar sizes.
/// Why it is needed: Connects standard primitive boundaries to internal compilation structures.
pub fn format_array_dty(dty: &DataTy) -> String {
    match &dty.dty {
        DataTyKind::Array(inner, nat) => {
            let inner_str = format_array_dty(inner);
            let size_str = nat_to_string(nat);
            format!("[{}; {}]", inner_str, size_str)
        }
        DataTyKind::Scalar(scalar) => match scalar {
            descend::ast::ScalarTy::U8 => "u8",
            descend::ast::ScalarTy::U32 => "u32",
            descend::ast::ScalarTy::U64 => "u64",
            descend::ast::ScalarTy::I32 => "i32",
            descend::ast::ScalarTy::I64 => "i64",
            descend::ast::ScalarTy::F32 => "f32",
            descend::ast::ScalarTy::F64 => "f64",
            descend::ast::ScalarTy::Bool => "bool",
            _ => "i32",
        }
        .to_string(),
        DataTyKind::Ident(ident) => ident.name.to_string(),
        DataTyKind::Struct(s) => s.ident.name.to_string(),
        DataTyKind::At(dty, _) => format_array_dty(dty),
        _ => "i32".to_string(),
    }
}

pub fn format_base_ty(ty: &Ty) -> String {
    if let TyKind::Data(dty) = &ty.ty {
        let mut current_dty = if let DataTyKind::Ref(ref_dty) = &dty.dty {
            &ref_dty.dty.dty
        } else {
            &dty.dty
        };
        while let DataTyKind::Array(elem_ty, _) = current_dty {
            current_dty = &elem_ty.dty;
        }
        if let DataTyKind::Scalar(scalar) = current_dty {
            return match scalar {
                descend::ast::ScalarTy::U8 => "u8",
                descend::ast::ScalarTy::U32 => "u32",
                descend::ast::ScalarTy::U64 => "u64",
                descend::ast::ScalarTy::I32 => "i32",
                descend::ast::ScalarTy::I64 => "i64",
                descend::ast::ScalarTy::F32 => "f32",
                descend::ast::ScalarTy::F64 => "f64",
                descend::ast::ScalarTy::Bool => "bool",
                _ => "i32",
            }
            .to_string();
        } else if let DataTyKind::Ident(ident) = current_dty {
            return ident.name.to_string();
        } else if let DataTyKind::Struct(s) = current_dty {
            return s.ident.name.to_string();
        }
    }
    "i32".to_string()
}

pub fn format_rust_struct(s: &descend::ast::StructDecl) -> String {
    let mut out = String::new();
    out.push_str("#[repr(C)]\n#[derive(Clone, Copy)]\npub struct ");
    out.push_str(&s.ident.name);
    out.push_str(" {\n");
    for (ident, dty) in &s.fields {
        out.push_str("    pub ");
        out.push_str(&ident.name);
        out.push_str(": ");
        out.push_str(&format_array_dty(dty));
        out.push_str(",\n");
    }
    out.push_str("}\n");
    out
}

/// Computes byte alignment constants for raw scalar variants.
///
/// What it does: Multiplies element types out by bytes.
/// Why it is needed: Needed explicitly for `cudarc` shared memory allocations.
pub fn extract_byte_size_from_ty(ty: &Ty) -> u32 {
    if let TyKind::Data(dty) = &ty.ty {
        let mut current_dty = if let DataTyKind::Ref(ref_dty) = &dty.dty {
            &ref_dty.dty.dty
        } else {
            &dty.dty
        };

        while let DataTyKind::Array(elem_ty, _) = current_dty {
            current_dty = &elem_ty.dty;
        }

        if let DataTyKind::Scalar(scalar) = current_dty {
            return match scalar {
                descend::ast::ScalarTy::U8 | descend::ast::ScalarTy::Bool => 1,
                descend::ast::ScalarTy::U32
                | descend::ast::ScalarTy::I32
                | descend::ast::ScalarTy::F32 => 4,
                descend::ast::ScalarTy::U64
                | descend::ast::ScalarTy::I64
                | descend::ast::ScalarTy::F64 => 8,
                _ => 4,
            };
        }
    }
    4 // fallback
}

/// Seeks within the caller stream to locate explicitly passed `call(...)` invocations to modify kernel generation.
///
/// What it does: Parses inner punctuation boundaries to extract arguments dynamically.
/// Why it is needed: Allows seamless function overriding during generic kernel implementations without rewriting tree nodes directly.
pub fn extract_call_args_tokens(stream: &mut TokenStream) -> Vec<String> {
    let mut call_args = Vec::new();
    let tokens: Vec<TokenTree> = stream.clone().into_iter().collect();

    let mut call_idx = None;
    for (i, tt) in tokens.iter().enumerate() {
        if let TokenTree::Ident(id) = tt {
            if id.to_string() == "call" {
                if i + 1 < tokens.len() {
                    if let TokenTree::Group(g) = &tokens[i + 1] {
                        if g.delimiter() == proc_macro::Delimiter::Parenthesis {
                            call_idx = Some(i);

                            // parse args from the group
                            let mut current_arg = String::new();
                            for arg_tt in g.stream() {
                                if let TokenTree::Punct(p) = &arg_tt {
                                    if p.as_char() == ',' {
                                        call_args.push(current_arg.trim().to_string());
                                        current_arg = String::new();
                                        continue;
                                    }
                                }
                                current_arg.push_str(&arg_tt.to_string());
                            }
                            if !current_arg.trim().is_empty() {
                                call_args.push(current_arg.trim().to_string());
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    if let Some(idx) = call_idx {
        // Remove `call(...)` from the stream
        let mut new_stream = TokenStream::new();
        for i in 0..idx {
            new_stream.extend(std::iter::once(tokens[i].clone()));
        }
        for i in idx + 2..tokens.len() {
            new_stream.extend(std::iter::once(tokens[i].clone()));
        }
        *stream = new_stream;
    }

    call_args
}
