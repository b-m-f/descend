use descend::ast::{AtomicTy, DataTy, DataTyKind, ScalarTy};

pub const UNSUPPORTED_TYPE: &str = "has a type that cannot cross to the host; only the basic \
    Descend types, arrays of them, and structs of them can";

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
    fn describe(self) -> (&'static str, Option<usize>) {
        match self {
            ScalarKind::Bool => ("bool", Some(1)),
            ScalarKind::U8 => ("u8", None),
            ScalarKind::U32 => ("u32", Some(4)),
            ScalarKind::U64 => ("u64", Some(8)),
            ScalarKind::I32 => ("i32", Some(4)),
            ScalarKind::I64 => ("i64", Some(8)),
            ScalarKind::F32 => ("f32", Some(4)),
            ScalarKind::F64 => ("f64", Some(8)),
            ScalarKind::AtomicU32 => ("u32", Some(4)),
            ScalarKind::AtomicI32 => ("i32", None),
        }
    }

    pub fn rust(self) -> &'static str {
        self.describe().0
    }

    pub fn shared_memory_bytes(self) -> Option<usize> {
        self.describe().1
    }
}

#[derive(Debug, Clone)]
pub struct ArraySize {
    elements: usize,
    total: usize,
}

#[derive(Debug, Clone)]
pub enum HostTy {
    Scalar(ScalarKind),
    Array(Box<HostTy>, ArraySize),
    Declared(String),
}

impl HostTy {
    pub fn rust(&self) -> String {
        match self {
            HostTy::Scalar(kind) => kind.rust().to_string(),
            HostTy::Array(element, array_size) => {
                format!("[{}; {}]", element.rust(), array_size.elements)
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
            _ => None,
        }
    }

    pub fn total_elements(&self) -> usize {
        match self {
            HostTy::Array(_, array_size) => array_size.total,
            HostTy::Scalar(_) | HostTy::Declared(_) => 1,
        }
    }
}

pub fn from_dty(dty: &DataTy) -> Result<HostTy, String> {
    match &dty.dty {
        DataTyKind::Scalar(scalar) => scalar_host_ty(*scalar),
        DataTyKind::Array(element, length_expression) => {
            let elements = crate::nat::eval_nat(length_expression)
                .map_err(|error| format!("has an unresolved size: {error}"))?;
            let element = from_dty(element)?;
            let total = element
                .total_elements()
                .checked_mul(elements)
                .ok_or_else(|| "has more elements than can be counted".to_string())?;
            Ok(HostTy::Array(
                Box::new(element),
                ArraySize { elements, total },
            ))
        }
        DataTyKind::Struct(declared) => Ok(HostTy::Declared(declared.ident.name.to_string())),
        DataTyKind::Ident(name) => Ok(HostTy::Declared(name.name.to_string())),
        DataTyKind::At(inner, _) => from_dty(inner),
        DataTyKind::Atomic(AtomicTy::AtomicU32) => Ok(HostTy::Scalar(ScalarKind::AtomicU32)),
        DataTyKind::Atomic(AtomicTy::AtomicI32) => Ok(HostTy::Scalar(ScalarKind::AtomicI32)),
        DataTyKind::Tuple(_) => Err(UNSUPPORTED_TYPE.to_string()),
        DataTyKind::ArrayShape(_, _) => Err(UNSUPPORTED_TYPE.to_string()),
        DataTyKind::Ref(_) => Err(UNSUPPORTED_TYPE.to_string()),
        DataTyKind::RawPtr(_) => Err(UNSUPPORTED_TYPE.to_string()),
        DataTyKind::Dead(_) => Err(UNSUPPORTED_TYPE.to_string()),
    }
}

fn scalar_host_ty(scalar: ScalarTy) -> Result<HostTy, String> {
    match scalar {
        ScalarTy::U8 => Ok(HostTy::Scalar(ScalarKind::U8)),
        ScalarTy::U32 => Ok(HostTy::Scalar(ScalarKind::U32)),
        ScalarTy::U64 => Ok(HostTy::Scalar(ScalarKind::U64)),
        ScalarTy::I32 => Ok(HostTy::Scalar(ScalarKind::I32)),
        ScalarTy::I64 => Ok(HostTy::Scalar(ScalarKind::I64)),
        ScalarTy::F32 => Ok(HostTy::Scalar(ScalarKind::F32)),
        ScalarTy::F64 => Ok(HostTy::Scalar(ScalarKind::F64)),
        ScalarTy::Bool => Ok(HostTy::Scalar(ScalarKind::Bool)),
        ScalarTy::Unit | ScalarTy::Gpu => Err(UNSUPPORTED_TYPE.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use descend::ast::Nat;

    fn scalar(scalar: ScalarTy) -> DataTy {
        DataTy::new(DataTyKind::Scalar(scalar))
    }

    fn array(element: DataTy, elements: usize) -> DataTy {
        DataTy::new(DataTyKind::Array(Box::new(element), Nat::Lit(elements)))
    }

    #[test]
    fn every_scalar_kind_states_its_rust_name_and_its_shared_memory_width_together() {
        for (kind, rust, shared_memory_bytes) in [
            (ScalarKind::Bool, "bool", Some(1)),
            (ScalarKind::U8, "u8", None),
            (ScalarKind::U32, "u32", Some(4)),
            (ScalarKind::U64, "u64", Some(8)),
            (ScalarKind::I32, "i32", Some(4)),
            (ScalarKind::I64, "i64", Some(8)),
            (ScalarKind::F32, "f32", Some(4)),
            (ScalarKind::F64, "f64", Some(8)),
            (ScalarKind::AtomicU32, "u32", Some(4)),
            (ScalarKind::AtomicI32, "i32", None),
        ] {
            assert_eq!(kind.describe(), (rust, shared_memory_bytes), "{kind:?}");
        }
    }

    #[test]
    fn every_scalar_type_reaches_a_rust_spelling() {
        for (scalar_ty, expected) in [
            (ScalarTy::U8, "u8"),
            (ScalarTy::U32, "u32"),
            (ScalarTy::U64, "u64"),
            (ScalarTy::I32, "i32"),
            (ScalarTy::I64, "i64"),
            (ScalarTy::F32, "f32"),
            (ScalarTy::F64, "f64"),
            (ScalarTy::Bool, "bool"),
        ] {
            assert_eq!(from_dty(&scalar(scalar_ty)).unwrap().rust(), expected);
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
        let host = from_dty(&dty).unwrap();
        assert_eq!(host.rust(), "[i32; 32]");
        assert_eq!(host.total_elements(), 32);
    }

    #[test]
    fn a_nested_array_nests_on_the_host_too() {
        let dty = array(array(scalar(ScalarTy::F32), 2), 3);
        let host = from_dty(&dty).unwrap();
        assert_eq!(host.rust(), "[[f32; 2]; 3]");
        assert_eq!(host.total_elements(), 6);
        assert_eq!(host.base().rust(), "f32");
    }

    #[test]
    fn an_array_size_wider_than_u32_is_carried_through_unchanged() {
        let wider_than_u32 = u32::MAX as usize + 1;
        let host = from_dty(&array(scalar(ScalarTy::I32), wider_than_u32)).unwrap();
        assert_eq!(host.rust(), "[i32; 4294967296]");
        assert_eq!(host.total_elements(), wider_than_u32);
    }

    #[test]
    fn a_declared_type_is_referred_to_by_name() {
        let dty = DataTy::new(DataTyKind::Ident(descend::ast::Ident::new("Pair")));
        assert_eq!(from_dty(&dty).unwrap().rust(), "Pair");
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
        assert_eq!(from_dty(&dty).unwrap().rust(), "i32");
    }

    #[test]
    fn every_unsupported_type_is_refused_with_the_same_sentence() {
        for case in [
            from_dty(&scalar(ScalarTy::Unit)),
            from_dty(&scalar(ScalarTy::Gpu)),
            from_dty(&DataTy::new(DataTyKind::Tuple(vec![
                scalar(ScalarTy::I32),
                scalar(ScalarTy::F32),
            ]))),
            from_dty(&DataTy::new(DataTyKind::ArrayShape(
                Box::new(scalar(ScalarTy::I32)),
                Nat::Lit(4),
            ))),
            from_dty(&DataTy::new(DataTyKind::RawPtr(Box::new(scalar(
                ScalarTy::I32,
            ))))),
            from_dty(&DataTy::new(DataTyKind::Dead(Box::new(scalar(
                ScalarTy::I32,
            ))))),
        ] {
            assert_eq!(case.unwrap_err(), UNSUPPORTED_TYPE);
        }
    }

    #[test]
    fn an_array_of_an_unsupported_element_is_refused_the_same_way() {
        let tuple = DataTy::new(DataTyKind::Tuple(vec![
            scalar(ScalarTy::I32),
            scalar(ScalarTy::F32),
        ]));
        assert_eq!(from_dty(&array(tuple, 64)).unwrap_err(), UNSUPPORTED_TYPE);
    }

    #[test]
    fn an_unresolved_size_says_so_rather_than_blaming_the_type() {
        let dty = DataTy::new(DataTyKind::Array(
            Box::new(scalar(ScalarTy::I32)),
            Nat::Ident(descend::ast::Ident::new("N")),
        ));
        let error = from_dty(&dty).unwrap_err();
        assert!(error.starts_with("has an unresolved size"), "{error}");
        assert!(error.contains("unknown identifier `N`"), "{error}");
    }

    #[test]
    fn an_array_too_large_to_count_is_refused() {
        let widest = u32::MAX as usize;
        let dty = array(array(array(scalar(ScalarTy::I32), widest), widest), widest);
        assert_eq!(
            from_dty(&dty).unwrap_err(),
            "has more elements than can be counted"
        );
    }
}
