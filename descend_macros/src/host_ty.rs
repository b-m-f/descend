//! How a Descend data type is represented on the host.
//!
//! This is the only place in the macro that reads `descend::ast::DataTyKind` for the
//! host boundary, and it does so with an **exhaustive** match. That is the point: a new
//! variant in Descend's AST becomes a compile error here, naming this one file, rather
//! than falling into a wildcard arm and producing a plausible-looking wrong answer.
//! `types::format_array_dty` shows what the wildcard costs — it answers `i32` for a
//! tuple, so a tuple field silently became a four-byte integer on the host.
//!
//! Everything the boundary needs is derived from [`HostTy`], so adding a representable
//! kind means extending the enum and this one match.

use descend::ast::{AtomicTy, DataTy, DataTyKind, Memory, Ownership, ScalarTy, Ty, TyKind};

/// A Descend scalar, with the spelling each side of the boundary uses for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    U8,
    U32,
    U64,
    I32,
    I64,
    F32,
    F64,
    Bool,
    AtomicU32,
    AtomicI32,
}

impl ScalarKind {
    /// How Rust spells it. An atomic is `u32`: the host only ever transfers its bytes,
    /// and the atomic operations live in the kernel.
    pub fn rust(self) -> &'static str {
        match self {
            ScalarKind::U8 => "u8",
            ScalarKind::U32 | ScalarKind::AtomicU32 => "u32",
            ScalarKind::U64 => "u64",
            ScalarKind::I32 | ScalarKind::AtomicI32 => "i32",
            ScalarKind::I64 => "i64",
            ScalarKind::F32 => "f32",
            ScalarKind::F64 => "f64",
            ScalarKind::Bool => "bool",
        }
    }

    /// How Descend spells it, for the `main` the macro synthesizes.
    pub fn descend(self) -> &'static str {
        match self {
            ScalarKind::U8 => "u8",
            ScalarKind::U32 => "u32",
            ScalarKind::U64 => "u64",
            ScalarKind::I32 => "i32",
            ScalarKind::I64 => "i64",
            ScalarKind::F32 => "f32",
            ScalarKind::F64 => "f64",
            ScalarKind::Bool => "bool",
            ScalarKind::AtomicU32 => "AtomicU32",
            ScalarKind::AtomicI32 => "AtomicI32",
        }
    }

    /// The width Descend's shared-memory sizing defines for this scalar, or `None`
    /// where Descend defines none.
    ///
    /// This mirrors `size_of_dty` in `src/codegen/mod.rs`, whose fallback arm is
    /// `panic!("unexpected data type")`. So `None` here is not a limitation the macro
    /// invents: it is the set of element types that would abort the Descend compiler.
    /// A shared `[u8; 64]` panics `descendc` outright, and a shared parameter of a
    /// declared type panics it in `get_elem_ty_and_amount`. Reporting those as
    /// diagnostics is the whole point of `check::check_shared_parameters`.
    ///
    /// `U8` and `AtomicI32` are therefore absent on purpose. `AtomicI32` is currently
    /// unreachable from Descend source anyway -- the grammar spells only `AtomicU32`
    /// (`src/parser/mod.rs:823`), so `AtomicI32` parses as a named type -- but if that
    /// spelling is ever added, this keeps the macro's answer and Descend's in step.
    ///
    /// See `tests/macro_compiler_safety/shared_param_unsupported_elem.rs`.
    pub fn shared_memory_bytes(self) -> Option<u32> {
        match self {
            ScalarKind::Bool => Some(1),
            ScalarKind::U32 | ScalarKind::I32 | ScalarKind::F32 | ScalarKind::AtomicU32 => Some(4),
            ScalarKind::U64 | ScalarKind::I64 | ScalarKind::F64 => Some(8),
            ScalarKind::U8 | ScalarKind::AtomicI32 => None,
        }
    }
}

/// An array length, in the two forms the two languages need.
///
/// Rust needs the value, to spell `[i32; 8192]` and to count elements. Descend needs
/// the expression the kernel wrote, because its type checker compares sizes
/// syntactically and `8192` does not unify with `64 * 128`.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayLen {
    value: u32,
    descend: String,
}

/// A Descend data type as the host can represent it.
#[derive(Debug, Clone, PartialEq)]
pub enum HostTy {
    Scalar(ScalarKind),
    Array(Box<HostTy>, ArrayLen),
    /// A type declared in the same `descend!` block, referred to by name.
    Declared(String),
}

impl HostTy {
    /// The Rust spelling of this type.
    pub fn rust(&self) -> String {
        match self {
            HostTy::Scalar(kind) => kind.rust().to_string(),
            HostTy::Array(element, array_length) => {
                format!("[{}; {}]", element.rust(), array_length.value)
            }
            HostTy::Declared(name) => name.clone(),
        }
    }

    /// The Descend spelling of this type.
    pub fn descend(&self) -> String {
        match self {
            HostTy::Scalar(kind) => kind.descend().to_string(),
            HostTy::Array(element, array_length) => {
                format!("[{}; {}]", element.descend(), array_length.descend)
            }
            HostTy::Declared(name) => name.clone(),
        }
    }

    /// The type of the innermost element, looking through any array nesting.
    pub fn base(&self) -> &HostTy {
        match self {
            HostTy::Array(element, _) => element.base(),
            HostTy::Scalar(_) | HostTy::Declared(_) => self,
        }
    }

    /// The scalar this type ultimately holds, if it holds one.
    pub fn scalar(&self) -> Option<ScalarKind> {
        match self.base() {
            HostTy::Scalar(kind) => Some(*kind),
            HostTy::Declared(_) => None,
            // `base` never returns an array.
            HostTy::Array(_, _) => None,
        }
    }

    /// How many elements of [`HostTy::base`] this type holds, or `None` if the count
    /// does not fit in a `u64`.
    pub fn elements(&self) -> Option<u64> {
        match self {
            HostTy::Array(element, array_length) => {
                u64::from(array_length.value).checked_mul(element.elements()?)
            }
            HostTy::Scalar(_) | HostTy::Declared(_) => Some(1),
        }
    }
}

/// What a reference parameter points at.
pub struct Reference<'a> {
    pub own: Ownership,
    pub mem: Memory,
    pub dty: &'a DataTy,
}

/// The reference a parameter is, if it is one.
///
/// Parameters reach a kernel either as a reference to memory or as a value, and that
/// distinction is read here so no caller has to match on `DataTyKind` to learn it.
pub fn reference(ty: &Ty) -> Option<Reference<'_>> {
    let TyKind::Data(dty) = &ty.ty else {
        return None;
    };
    // A wildcard is right here, unlike in `host_ty`: the question is only whether this
    // is a reference, and a data type Descend adds later is simply not one.
    match &dty.dty {
        DataTyKind::Ref(reference_dty) => Some(Reference {
            own: reference_dty.own,
            mem: reference_dty.mem.clone(),
            dty: &reference_dty.dty,
        }),
        _ => None,
    }
}

/// The data type a parameter passes by value, if it passes one.
pub fn value(ty: &Ty) -> Option<&DataTy> {
    let TyKind::Data(dty) = &ty.ty else {
        return None;
    };
    match &dty.dty {
        DataTyKind::Ref(_) => None,
        _ => Some(dty.as_ref()),
    }
}

/// The data a parameter names: what a reference points at, or the value itself.
pub fn parameter_dty(ty: &Ty) -> Option<&DataTy> {
    reference(ty).map(|r| r.dty).or_else(|| value(ty))
}

/// Why a Descend type has no host representation.
///
/// The text completes a sentence naming what was rejected, so one reason serves both
/// `struct `P`: field `f` ..` and `kernel `k`: parameter `p` ..`.
#[derive(Debug)]
pub struct Unrepresentable(String);

impl Unrepresentable {
    pub fn reason(&self) -> &str {
        &self.0
    }
}

/// The host representation of a Descend data type, or why it has none.
pub fn host_ty(dty: &DataTy) -> Result<HostTy, Unrepresentable> {
    match &dty.dty {
        DataTyKind::Scalar(scalar) => scalar_host_ty(*scalar),
        DataTyKind::Array(element, length_expression) => {
            let array_length = ArrayLen {
                value: crate::nat::eval_nat(length_expression)
                    .map_err(|error| Unrepresentable(format!("has an unresolved size: {error}")))?,
                descend: crate::spell::format_nat(length_expression)
                    .map_err(|error| Unrepresentable(format!("has an unresolved size: {error}")))?,
            };
            // Carry the element's reason outward, so a rejection names the type the
            // reader actually wrote rather than the element buried inside it.
            let element = host_ty(element).map_err(|reason| {
                Unrepresentable(format!("is an array whose element {}", reason.reason()))
            })?;
            Ok(HostTy::Array(Box::new(element), array_length))
        }
        DataTyKind::Struct(declared) => Ok(HostTy::Declared(declared.ident.name.to_string())),
        DataTyKind::Ident(name) => Ok(HostTy::Declared(name.name.to_string())),
        // A memory annotation says where the value lives, not what it is.
        DataTyKind::At(inner, _) => host_ty(inner),
        DataTyKind::Tuple(_) => Err(Unrepresentable(
            "is a tuple. Neither Rust tuples nor `descend::tuple` guarantee a memory \
             layout, so a tuple cannot cross to the host; group the values in a Descend \
             struct instead. Tuples inside a kernel are unaffected."
                .to_string(),
        )),
        DataTyKind::ArrayShape(_, _) => Err(Unrepresentable(
            "is an array view. A view describes how a kernel reads memory, so it has no \
             host value; pass the array it views instead."
                .to_string(),
        )),
        DataTyKind::Atomic(AtomicTy::AtomicU32) => Ok(HostTy::Scalar(ScalarKind::AtomicU32)),
        DataTyKind::Atomic(AtomicTy::AtomicI32) => Ok(HostTy::Scalar(ScalarKind::AtomicI32)),
        DataTyKind::Ref(_) => Err(Unrepresentable(
            "is a reference. A reference points into GPU memory, which the host cannot \
             hold directly; it reaches the kernel as a buffer argument."
                .to_string(),
        )),
        DataTyKind::RawPtr(_) => Err(Unrepresentable(
            "is a raw pointer, which carries no length and so cannot be transferred \
             safely."
                .to_string(),
        )),
        DataTyKind::Dead(_) => Err(Unrepresentable(
            "has been moved out of, so it holds no value to transfer.".to_string(),
        )),
    }
}

fn scalar_host_ty(scalar: ScalarTy) -> Result<HostTy, Unrepresentable> {
    match scalar {
        ScalarTy::U8 => Ok(HostTy::Scalar(ScalarKind::U8)),
        ScalarTy::U32 => Ok(HostTy::Scalar(ScalarKind::U32)),
        ScalarTy::U64 => Ok(HostTy::Scalar(ScalarKind::U64)),
        ScalarTy::I32 => Ok(HostTy::Scalar(ScalarKind::I32)),
        ScalarTy::I64 => Ok(HostTy::Scalar(ScalarKind::I64)),
        ScalarTy::F32 => Ok(HostTy::Scalar(ScalarKind::F32)),
        ScalarTy::F64 => Ok(HostTy::Scalar(ScalarKind::F64)),
        ScalarTy::Bool => Ok(HostTy::Scalar(ScalarKind::Bool)),
        ScalarTy::Unit => Err(Unrepresentable(
            "is the unit type, which holds nothing to transfer.".to_string(),
        )),
        ScalarTy::Gpu => Err(Unrepresentable(
            "is a GPU handle, which belongs to the context rather than to a buffer.".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use descend::ast::Nat;

    fn scalar(scalar: ScalarTy) -> DataTy {
        DataTy::new(DataTyKind::Scalar(scalar))
    }

    #[test]
    fn every_scalar_width_has_a_rust_spelling() {
        for (ty, expected) in [
            (ScalarTy::U8, "u8"),
            (ScalarTy::U32, "u32"),
            (ScalarTy::U64, "u64"),
            (ScalarTy::I32, "i32"),
            (ScalarTy::I64, "i64"),
            (ScalarTy::F32, "f32"),
            (ScalarTy::F64, "f64"),
            (ScalarTy::Bool, "bool"),
        ] {
            assert_eq!(host_ty(&scalar(ty)).unwrap().rust(), expected);
        }
    }

    #[test]
    fn an_array_resolves_its_length() {
        let dty = DataTy::new(DataTyKind::Array(
            Box::new(scalar(ScalarTy::I32)),
            Nat::BinOp(
                descend::ast::BinOpNat::Mul,
                Box::new(Nat::Lit(4)),
                Box::new(Nat::Lit(8)),
            ),
        ));
        let host = host_ty(&dty).unwrap();
        // Rust gets the value; Descend keeps the expression, because its type checker
        // compares sizes syntactically.
        assert_eq!(host.rust(), "[i32; 32]");
        assert_eq!(host.descend(), "[i32; 4 * 8]");
        assert_eq!(host.elements(), Some(32));
    }

    #[test]
    fn a_nested_array_nests_on_the_host_too() {
        let inner = DataTy::new(DataTyKind::Array(
            Box::new(scalar(ScalarTy::F32)),
            Nat::Lit(2),
        ));
        let dty = DataTy::new(DataTyKind::Array(Box::new(inner), Nat::Lit(3)));
        assert_eq!(host_ty(&dty).unwrap().rust(), "[[f32; 2]; 3]");
    }

    #[test]
    fn a_declared_type_is_referred_to_by_name() {
        let dty = DataTy::new(DataTyKind::Ident(descend::ast::Ident::new("Pair")));
        assert_eq!(host_ty(&dty).unwrap().rust(), "Pair");
    }

    #[test]
    fn a_tuple_is_rejected_with_the_layout_as_the_reason() {
        let dty = DataTy::new(DataTyKind::Tuple(vec![
            scalar(ScalarTy::F32),
            scalar(ScalarTy::F32),
        ]));
        let reason = host_ty(&dty).unwrap_err();
        assert!(reason.reason().contains("tuple"), "{reason:?}");
        assert!(reason.reason().contains("memory layout"), "{reason:?}");
    }

    #[test]
    fn each_unrepresentable_kind_says_why_rather_than_merely_refusing() {
        let cases = [
            host_ty(&scalar(ScalarTy::Unit)),
            host_ty(&scalar(ScalarTy::Gpu)),
            host_ty(&DataTy::new(DataTyKind::ArrayShape(
                Box::new(scalar(ScalarTy::I32)),
                Nat::Lit(4),
            ))),
            host_ty(&DataTy::new(DataTyKind::RawPtr(Box::new(scalar(
                ScalarTy::I32,
            ))))),
        ];
        for case in cases {
            let reason = case.expect_err("must be rejected").reason().to_string();
            // A reason, not just a name: it has to tell the reader what to do instead or
            // why the type has no host value at all.
            assert!(reason.ends_with('.'), "not a sentence: {reason}");
            assert!(reason.len() > 40, "too terse to act on: {reason}");
        }
    }

    #[test]
    fn an_array_of_an_unrepresentable_element_says_so() {
        let dty = DataTy::new(DataTyKind::Array(
            Box::new(DataTy::new(DataTyKind::Tuple(vec![
                scalar(ScalarTy::I32),
                scalar(ScalarTy::F32),
            ]))),
            Nat::Lit(64),
        ));
        let reason = host_ty(&dty).unwrap_err();
        assert!(
            reason
                .reason()
                .starts_with("is an array whose element is a tuple"),
            "{reason:?}"
        );
    }

    /// Descend's `size_of_dty` panics on an element type it has no width for, so the
    /// macro must answer `None` for exactly those and no others. Verified against
    /// `descendc`: a shared `[u8; 64]` aborts it, a shared `[i32; 64]` yields
    /// `(0 + (4 * (1 * 64)))` bytes.
    #[test]
    fn shared_memory_widths_mirror_what_descend_can_size() {
        for (kind, expected) in [
            (ScalarKind::Bool, Some(1)),
            (ScalarKind::U32, Some(4)),
            (ScalarKind::I32, Some(4)),
            (ScalarKind::F32, Some(4)),
            (ScalarKind::AtomicU32, Some(4)),
            (ScalarKind::U64, Some(8)),
            (ScalarKind::I64, Some(8)),
            (ScalarKind::F64, Some(8)),
            // Descend panics on these rather than sizing them.
            (ScalarKind::U8, None),
            (ScalarKind::AtomicI32, None),
        ] {
            assert_eq!(kind.shared_memory_bytes(), expected, "{kind:?}");
        }
    }

    #[test]
    fn a_declared_type_has_no_shared_memory_width() {
        // Descend panics on a shared parameter of a declared type, so the macro must
        // not offer a width for one.
        assert_eq!(HostTy::Declared("Pair".to_string()).scalar(), None);
    }

    #[test]
    fn a_memory_annotation_does_not_change_the_representation() {
        let dty = DataTy::new(DataTyKind::At(
            Box::new(scalar(ScalarTy::I32)),
            descend::ast::Memory::GpuGlobal,
        ));
        assert_eq!(host_ty(&dty).unwrap().rust(), "i32");
    }
}
