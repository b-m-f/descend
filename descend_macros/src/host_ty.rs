use descend::ast::{AtomicTy, DataTy, DataTyKind, Memory, Ownership, ScalarTy, Ty, TyKind};

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

    pub fn shared_memory_bytes(self) -> Option<u32> {
        match self {
            ScalarKind::Bool => Some(1),
            ScalarKind::U32 | ScalarKind::I32 | ScalarKind::F32 | ScalarKind::AtomicU32 => Some(4),
            ScalarKind::U64 | ScalarKind::I64 | ScalarKind::F64 => Some(8),
            ScalarKind::U8 | ScalarKind::AtomicI32 => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayLen {
    value: u32,
    descend: String,
}

impl ArrayLen {
    pub(crate) fn new(value: u32, descend: impl Into<String>) -> Self {
        Self {
            value,
            descend: descend.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum HostTy {
    Scalar(ScalarKind),
    Array(Box<HostTy>, ArrayLen),
    Declared(String),
}

impl HostTy {
    pub fn rust(&self) -> String {
        match self {
            HostTy::Scalar(kind) => kind.rust().to_string(),
            HostTy::Array(element, array_length) => {
                format!("[{}; {}]", element.rust(), array_length.value)
            }
            HostTy::Declared(name) => name.clone(),
        }
    }

    pub fn base(&self) -> &HostTy {
        match self {
            HostTy::Array(element, _) => element.base(),
            HostTy::Scalar(_) | HostTy::Declared(_) => self,
        }
    }

    pub fn scalar(&self) -> Option<ScalarKind> {
        match self.base() {
            HostTy::Scalar(kind) => Some(*kind),
            HostTy::Declared(_) => None,
            HostTy::Array(_, _) => None,
        }
    }

    pub fn elements(&self) -> Option<u64> {
        match self {
            HostTy::Array(element, array_length) => {
                u64::from(array_length.value).checked_mul(element.elements()?)
            }
            HostTy::Scalar(_) | HostTy::Declared(_) => Some(1),
        }
    }
}

pub struct Reference<'a> {
    pub ownership: Ownership,
    pub memory: Memory,
    pub dty: &'a DataTy,
}

pub fn reference(ty: &Ty) -> Option<Reference<'_>> {
    let TyKind::Data(dty) = &ty.ty else {
        return None;
    };
    match &dty.dty {
        DataTyKind::Ref(reference_dty) => Some(Reference {
            ownership: reference_dty.own,
            memory: reference_dty.mem.clone(),
            dty: &reference_dty.dty,
        }),
        _ => None,
    }
}

pub fn value(ty: &Ty) -> Option<&DataTy> {
    let TyKind::Data(dty) = &ty.ty else {
        return None;
    };
    match &dty.dty {
        DataTyKind::Ref(_) => None,
        _ => Some(dty.as_ref()),
    }
}

pub fn parameter_dty(ty: &Ty) -> Option<&DataTy> {
    reference(ty).map(|r| r.dty).or_else(|| value(ty))
}

#[derive(Debug)]
pub struct Unrepresentable(String);

impl Unrepresentable {
    pub fn reason(&self) -> &str {
        &self.0
    }
}

pub fn host_ty(dty: &DataTy) -> Result<HostTy, Unrepresentable> {
    match &dty.dty {
        DataTyKind::Scalar(scalar) => scalar_host_ty(*scalar),
        DataTyKind::Array(element, length_expression) => {
            let array_length = ArrayLen::new(
                crate::nat::eval_nat(length_expression)
                    .map_err(|error| Unrepresentable(format!("has an unresolved size: {error}")))?,
                crate::spell::format_nat(length_expression)
                    .map_err(|error| Unrepresentable(format!("has an unresolved size: {error}")))?,
            );
            let element = host_ty(element).map_err(|reason| {
                Unrepresentable(format!("is an array whose element {}", reason.reason()))
            })?;
            Ok(HostTy::Array(Box::new(element), array_length))
        }
        DataTyKind::Struct(declared) => Ok(HostTy::Declared(declared.ident.name.to_string())),
        DataTyKind::Ident(name) => Ok(HostTy::Declared(name.name.to_string())),
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
        assert_eq!(host.rust(), "[i32; 32]");
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
            (ScalarKind::U8, None),
            (ScalarKind::AtomicI32, None),
        ] {
            assert_eq!(kind.shared_memory_bytes(), expected, "{kind:?}");
        }
    }

    #[test]
    fn a_declared_type_has_no_shared_memory_width() {
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
