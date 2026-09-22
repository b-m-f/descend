#![allow(unreachable_patterns)]
use descend::ast::*;
use proc_macro2::{TokenStream, TokenTree};
use std::collections::HashMap;

/// Parses a `<N: 16384, bs: 16>` block of generic bindings.
///
/// Both a launch site and an instantiation name a generic kernel's `nat`
/// parameters this way, and both need them as literals: they are substituted
/// into the Descend source before it is compiled.
pub fn parse_generic_bindings(input: TokenStream) -> Result<HashMap<String, String>, String> {
    let mut bindings = HashMap::new();
    let mut tokens = input.into_iter().peekable();

    match tokens.next() {
        Some(TokenTree::Punct(p)) if p.as_char() == '<' => {}
        None => return Ok(bindings),
        Some(other) => {
            return Err(format!(
                "expected `<` to open the generic bindings, found `{other}`"
            ))
        }
    }

    loop {
        match tokens.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == '>' => break,
            Some(TokenTree::Punct(p)) if p.as_char() == ',' => continue,
            Some(TokenTree::Ident(name)) => {
                match tokens.next() {
                    Some(TokenTree::Punct(p)) if p.as_char() == ':' => {}
                    _ => return Err(format!("expected `:` after generic `{name}`")),
                }
                match tokens.next() {
                    Some(TokenTree::Literal(value)) => {
                        bindings.insert(name.to_string(), value.to_string());
                    }
                    _ => {
                        return Err(format!(
                            "generic `{name}` needs an integer literal: a kernel's sizes are \
                             compiled into its code, so they cannot come from a runtime value"
                        ))
                    }
                }
            }
            Some(other) => return Err(format!("unexpected `{other}` in the generic bindings")),
            None => return Err("the generic bindings are not terminated by `>`".to_string()),
        }
    }

    Ok(bindings)
}

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

/// The type of a parameter the kernel takes by value, as both Descend and Rust
/// spell it: `i32`, `[i32; 3]`, or a struct name. Empty for a reference
/// parameter, which is passed as a device address instead.
pub fn format_value_ty(ty: &Ty) -> String {
    if let TyKind::Data(dty) = &ty.ty {
        if !matches!(dty.dty, DataTyKind::Ref(_)) {
            return format_array_dty(dty);
        }
    }
    String::new()
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

/// Renders a data type with every array size evaluated against `bindings`.
///
/// The text form of a type is produced once, here, from the AST — rather than
/// produced early and patched afterwards, which is how a size name inside an
/// unrelated position gets rewritten.
pub fn format_dty_eval(dty: &DataTy, bindings: &HashMap<String, String>) -> Result<String, String> {
    match &dty.dty {
        DataTyKind::Array(inner, nat) => {
            let size = crate::nat::eval_nat(nat, bindings)?;
            Ok(format!("[{}; {}]", format_dty_eval(inner, bindings)?, size))
        }
        DataTyKind::At(inner, _) => format_dty_eval(inner, bindings),
        _ => Ok(format_array_dty(dty)),
    }
}

/// Returns the Rust element type of a kernel parameter, or `None` when the
/// Descend element type has no supported host representation.
pub fn format_base_ty(ty: &Ty) -> Option<String> {
    let TyKind::Data(dty) = &ty.ty else {
        return None;
    };
    let mut current_dty = if let DataTyKind::Ref(ref_dty) = &dty.dty {
        &ref_dty.dty.dty
    } else {
        &dty.dty
    };
    while let DataTyKind::Array(elem_ty, _) = current_dty {
        current_dty = &elem_ty.dty;
    }
    match current_dty {
        DataTyKind::Scalar(scalar) => match scalar {
            ScalarTy::U8 => Some("u8"),
            ScalarTy::U32 => Some("u32"),
            ScalarTy::U64 => Some("u64"),
            ScalarTy::I32 => Some("i32"),
            ScalarTy::I64 => Some("i64"),
            ScalarTy::F32 => Some("f32"),
            ScalarTy::F64 => Some("f64"),
            ScalarTy::Bool => Some("bool"),
            ScalarTy::Unit | ScalarTy::Gpu => None,
        }
        .map(str::to_string),
        DataTyKind::Ident(ident) => Some(ident.name.to_string()),
        DataTyKind::Struct(s) => Some(s.ident.name.to_string()),
        _ => None,
    }
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

/// The byte size of a parameter's element type, or `None` when the Descend code
/// generator does not define one.
///
/// This mirrors `size_of_dty` in `descend::codegen` exactly, because the host's
/// dynamic shared-memory request must agree with the packing the generated CUDA
/// performs. In particular `u8` is *not* sized here: the code generator panics
/// on it, so a `u8` shared parameter has to be rejected with a message rather
/// than given a plausible-looking size.
pub fn elem_byte_size(ty: &Ty) -> Option<u32> {
    let TyKind::Data(dty) = &ty.ty else {
        return None;
    };
    let mut current_dty = if let DataTyKind::Ref(ref_dty) = &dty.dty {
        &ref_dty.dty.dty
    } else {
        &dty.dty
    };
    while let DataTyKind::Array(elem_ty, _) = current_dty {
        current_dty = &elem_ty.dty;
    }
    match current_dty {
        DataTyKind::Scalar(ScalarTy::Bool) => Some(1),
        DataTyKind::Scalar(ScalarTy::U32 | ScalarTy::I32 | ScalarTy::F32) => Some(4),
        DataTyKind::Atomic(descend::ast::AtomicTy::AtomicU32) => Some(4),
        DataTyKind::Scalar(ScalarTy::U64 | ScalarTy::I64 | ScalarTy::F64) => Some(8),
        _ => None,
    }
}

/// The array dimensions of a parameter's type, outermost first.
///
/// Empty for a parameter that is not an array. Unlike [`extract_size_from_ty`],
/// which renders the dimensions as text for substitution into generated source,
/// this keeps the `Nat`s so they can be evaluated against the launch's generic
/// bindings.
pub fn extract_dim_nats(ty: &Ty) -> Vec<Nat> {
    let TyKind::Data(dty) = &ty.ty else {
        return Vec::new();
    };
    let mut current_dty = if let DataTyKind::Ref(ref_dty) = &dty.dty {
        &ref_dty.dty.dty
    } else {
        &dty.dty
    };
    let mut dims = Vec::new();
    while let DataTyKind::Array(elem_ty, nat) = current_dty {
        dims.push(nat.clone());
        current_dty = &elem_ty.dty;
    }
    dims
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn bindings(source: &str) -> Result<HashMap<String, String>, String> {
        parse_generic_bindings(TokenStream::from_str(source).expect("test input must tokenize"))
    }

    #[test]
    fn reads_several_sizes() {
        let parsed = bindings("<bs: 16, m: 64, n: 128, k: 64>").unwrap();
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed["bs"], "16");
        assert_eq!(parsed["k"], "64");
    }

    #[test]
    fn an_empty_binding_list_is_allowed() {
        assert!(bindings("<>").unwrap().is_empty());
        assert!(bindings("").unwrap().is_empty());
    }

    #[test]
    fn a_runtime_expression_is_rejected_with_the_reason() {
        let error = bindings("<N: n>").unwrap_err();
        assert!(
            error.contains("needs an integer literal") && error.contains("compiled into its code"),
            "unhelpful error: {error}"
        );
    }

    #[test]
    fn a_missing_colon_is_rejected() {
        assert!(bindings("<N 16384>").unwrap_err().contains("expected `:`"));
    }

    #[test]
    fn an_unterminated_list_is_rejected() {
        assert!(bindings("<N: 16384")
            .unwrap_err()
            .contains("not terminated"));
    }
}
