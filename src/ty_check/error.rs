use super::Ty;
use crate::ast::internal::Place;
use crate::ast::printer::PrintState;
use crate::ast::{BaseExec, DataTy, Expr, Ident, NatEvalError, Ownership, PlaceExpr, Span, TyKind};
use crate::error;
use crate::error::{default_format, ErrorReported};
use crate::parser::SourceCode;
use annotate_snippets::display_list::DisplayList;
use annotate_snippets::snippet::{Annotation, AnnotationType, Snippet};

#[must_use]
#[derive(Debug)]
pub enum TyError {
    MultiError(Vec<TyError>),
    MutabilityNotAllowed(Ty),
    CtxError(CtxError),
    SubTyError(SubTyError),
    // Standard data type mismatch, expected type followed by actual type
    MismatchedDataTypes(DataTy, DataTy, Expr),
    // "Trying to violate existing borrow of {:?}.",
    // p1 under own1 is in conflict because of BorrowingError
    ConflictingBorrow(Box<PlaceExpr>, Ownership, BorrowingError),
    PrvValueAlreadyInUse(String),
    // No loan the reference points to has a type that fits the reference element type
    ReferenceToIncompatibleType,
    // ownership of reference and loan it refers to do not fit
    ReferenceToWrongOwnership,
    // This would mean that the reference points to nothing, e.g., because the value was moved
    // out from under the reference which is forbidden.
    ReferenceToDeadTy,
    // Assignment to a constant place expression.
    AssignToConst(PlaceExpr), //, Box<Expr>),
    // Assigning to a view is forbidden
    AssignToView,
    // Trying to split a non-view array.
    SplittingNonViewArray,
    // Expected a different type
    ExpectedTupleType(TyKind, PlaceExpr),
    // Trying to borrow uniquely but place is not mutable
    ConstBorrow(PlaceExpr),
    CannotUniquelyDereferenceShared(Option<Span>),
    // The borrowed view type is at least paritally dead
    BorrowingDeadView,
    IllegalExec,
    // Trying to type an expression with dead type
    DeadTy,
    // When a parallel collection consits of other parallel elements, a for-with requires an
    // identifier for these elements.
    MissingParallelCollectionIdent,
    // If a provenance place holder is not substituted for a real provenance
    CouldNotInferProvenance,
    // The annotated or inferred type of the pattern does not fit the pattern.
    PatternAndTypeDoNotMatch,
    UnexpectedType,
    // The thread hierarchy dimension referred to does not exist
    IllegalDimension,
    UnifyError(UnifyError),
    MissingMain,
    NatEvalError(NatEvalError),
    CannotInferGenericArg(Ident),
    UnsafeRequired,
    // TODO remove as soon as possible
    String(String),
}

impl FromIterator<TyError> for TyError {
    fn from_iter<T: IntoIterator<Item = TyError>>(iter: T) -> Self {
        TyError::MultiError(iter.into_iter().collect())
    }
}

impl TyError {
    pub fn span(&self) -> Option<Span> {
        match self {
            TyError::MultiError(errors) => errors.iter().find_map(TyError::span),
            TyError::MutabilityNotAllowed(ty) => ty.span,
            TyError::CtxError(error) => error.span(),
            TyError::SubTyError(error) => error.span(),
            TyError::MismatchedDataTypes(_, _, expr) => expr.span,
            TyError::ConflictingBorrow(place, _, error) => place.span.or_else(|| error.span()),
            TyError::AssignToConst(place) | TyError::ConstBorrow(place) => place.span,
            TyError::CannotUniquelyDereferenceShared(span) => *span,
            TyError::ExpectedTupleType(_, place) => place.span,
            TyError::UnifyError(error) => error.span(),
            TyError::CannotInferGenericArg(ident) => ident.span,
            _ => None,
        }
    }

    pub fn message(&self) -> String {
        match self {
            TyError::MultiError(_) => "multiple type errors".to_string(),
            TyError::MutabilityNotAllowed(_) => "mutability is not allowed here".to_string(),
            TyError::CtxError(error) => error.message(),
            TyError::SubTyError(error) => error.message(),
            TyError::MismatchedDataTypes(expected, actual, _) => {
                let mut expected_printer = PrintState::new();
                expected_printer.print_dty(expected);
                let mut actual_printer = PrintState::new();
                actual_printer.print_dty(actual);
                format!(
                    "mismatched types: expected `{}` but found `{}`",
                    expected_printer.get(),
                    actual_printer.get()
                )
            }
            TyError::ConflictingBorrow(_, _, error) => error.message(),
            TyError::PrvValueAlreadyInUse(name) => {
                format!("provenance `{name}` is already in use")
            }
            TyError::ReferenceToIncompatibleType => {
                "reference points to a value with an incompatible type".to_string()
            }
            TyError::ReferenceToWrongOwnership => {
                "reference ownership does not match its loan".to_string()
            }
            TyError::ReferenceToDeadTy => "reference points to a dead value".to_string(),
            TyError::AssignToConst(_) => "cannot assign to a constant place".to_string(),
            TyError::AssignToView => "cannot assign to a view".to_string(),
            TyError::SplittingNonViewArray => {
                "cannot split an array that is not a view".to_string()
            }
            TyError::ExpectedTupleType(_, _) => "expected a tuple type".to_string(),
            TyError::ConstBorrow(_) => "cannot borrow a constant place uniquely".to_string(),
            TyError::CannotUniquelyDereferenceShared(_) => {
                "cannot uniquely access through a shared reference".to_string()
            }
            TyError::BorrowingDeadView => "cannot borrow a dead view".to_string(),
            TyError::IllegalExec => "illegal execution expression".to_string(),
            TyError::DeadTy => "cannot use a value with a dead type".to_string(),
            TyError::MissingParallelCollectionIdent => {
                "parallel collection requires an element identifier".to_string()
            }
            TyError::CouldNotInferProvenance => "could not infer provenance".to_string(),
            TyError::PatternAndTypeDoNotMatch => "pattern and type do not match".to_string(),
            TyError::UnexpectedType => "unexpected type".to_string(),
            TyError::IllegalDimension => "execution dimension does not exist".to_string(),
            TyError::UnifyError(error) => error.message(),
            TyError::MissingMain => "missing `main` function".to_string(),
            TyError::NatEvalError(_) => "could not evaluate a natural number".to_string(),
            TyError::CannotInferGenericArg(ident) => {
                format!("could not infer generic argument `{}`", ident.name)
            }
            TyError::UnsafeRequired => "unsafe operation requires an unsafe context".to_string(),
            TyError::String(message) => message.clone(),
        }
    }

    pub fn emit(&self, source: &SourceCode) -> ErrorReported {
        if let TyError::MultiError(errors) = self {
            for error in errors {
                error.emit(source);
            }
            return ErrorReported;
        }
        let message = self.message();
        let snippet = self.span().and_then(|span| {
            let (begin_line, begin_column) = source.get_line_col(span.begin);
            let (end_line, end_column) = source.get_line_col(span.end);
            (begin_line == end_line).then(|| {
                error::single_line_snippet(
                    source,
                    &message,
                    &message,
                    begin_line,
                    begin_column,
                    end_column.max(begin_column + 1),
                )
            })
        });
        let snippet = snippet.unwrap_or_else(|| Snippet {
            title: Some(Annotation {
                id: None,
                label: Some(&message),
                annotation_type: AnnotationType::Error,
            }),
            footer: vec![],
            slices: vec![],
            opt: default_format(),
        });
        eprintln!("{}", DisplayList::from(snippet));
        ErrorReported
    }
}

impl From<CtxError> for TyError {
    fn from(err: CtxError) -> Self {
        TyError::CtxError(err)
    }
}
impl From<SubTyError> for TyError {
    fn from(err: SubTyError) -> Self {
        TyError::SubTyError(err)
    }
}
impl From<UnifyError> for TyError {
    fn from(err: UnifyError) -> Self {
        TyError::UnifyError(err)
    }
}
impl From<NatEvalError> for TyError {
    fn from(err: NatEvalError) -> Self {
        TyError::NatEvalError(err)
    }
}

#[must_use]
#[derive(Debug)]
pub enum SubTyError {
    CtxError(CtxError),
    // format!("{} lives longer than {}.", shorter, longer)
    NotOutliving(String, String),
    // format!("No loans bound to provenance.")
    PrvNotUsedInBorrow(String),
    // Subtyping checks fail if the memory kinds are not equal
    MemoryKindsNoMatch,
    // Subtyping checks fail if the ownership of supposedly subtyped references do not match
    OwnershipNoMatch,
    SizesNoMatch,
    // TODO remove asap
    Dummy,
}

impl SubTyError {
    fn span(&self) -> Option<Span> {
        match self {
            SubTyError::CtxError(error) => error.span(),
            _ => None,
        }
    }

    fn message(&self) -> String {
        match self {
            SubTyError::CtxError(error) => error.message(),
            SubTyError::NotOutliving(shorter, longer) => {
                format!("`{shorter}` does not outlive `{longer}`")
            }
            SubTyError::PrvNotUsedInBorrow(name) => {
                format!("no loan is bound to provenance `{name}`")
            }
            SubTyError::MemoryKindsNoMatch => "memory kinds do not match".to_string(),
            SubTyError::OwnershipNoMatch => "reference ownership does not match".to_string(),
            SubTyError::SizesNoMatch => "array sizes do not match".to_string(),
            SubTyError::Dummy => "subtyping failed".to_string(),
        }
    }
}

#[must_use]
#[derive(Debug)]
pub enum UnifyError {
    // Cannot unify the two terms
    CannotUnify,
    CannotUnifyTy(Ty, Ty),
    CannotUnifyDataTy(DataTy, DataTy),
    // A type variable has to be equal to a term that is referring to the same type variable
    InfiniteType,
    SubTyError(SubTyError),
}

impl UnifyError {
    fn span(&self) -> Option<Span> {
        match self {
            UnifyError::CannotUnifyTy(left, right) => left.span.or(right.span),
            UnifyError::CannotUnifyDataTy(left, right) => left.span.or(right.span),
            UnifyError::SubTyError(error) => error.span(),
            _ => None,
        }
    }

    fn message(&self) -> String {
        match self {
            UnifyError::CannotUnify => "types cannot be unified".to_string(),
            UnifyError::CannotUnifyTy(_, _) | UnifyError::CannotUnifyDataTy(_, _) => {
                "types cannot be unified".to_string()
            }
            UnifyError::InfiniteType => "type contains itself".to_string(),
            UnifyError::SubTyError(error) => error.message(),
        }
    }
}

impl From<SubTyError> for UnifyError {
    fn from(err: SubTyError) -> Self {
        UnifyError::SubTyError(err)
    }
}

#[must_use]
#[derive(Debug)]
pub enum CtxError {
    //format!("Identifier: {} not found in context.", ident)),
    IdentNotFound(Ident),
    //"Cannot find identifier {} in kinding context",
    KindedIdentNotFound(Ident),
    // "Typing Context is missing the provenance value {}",
    PrvValueNotFound(String),
    // format!("{} is not declared", prv_rel.longer));
    PrvIdentNotFound(Ident),
    // format!("{} is not defined as outliving {}.", l, s)
    OutlRelNotDefined(Ident, Ident),
    // TODO move to TyError
    IllegalProjection,
}

impl CtxError {
    fn span(&self) -> Option<Span> {
        match self {
            CtxError::IdentNotFound(ident)
            | CtxError::KindedIdentNotFound(ident)
            | CtxError::PrvIdentNotFound(ident) => ident.span,
            CtxError::OutlRelNotDefined(left, right) => left.span.or(right.span),
            _ => None,
        }
    }

    fn message(&self) -> String {
        match self {
            CtxError::IdentNotFound(ident) => {
                format!("identifier `{}` was not found", ident.name)
            }
            CtxError::KindedIdentNotFound(ident) => {
                format!("generic identifier `{}` was not found", ident.name)
            }
            CtxError::PrvValueNotFound(name) => {
                format!("provenance value `{name}` was not found")
            }
            CtxError::PrvIdentNotFound(ident) => {
                format!("provenance `{}` was not declared", ident.name)
            }
            CtxError::OutlRelNotDefined(longer, shorter) => {
                format!(
                    "`{}` is not defined as outliving `{}`",
                    longer.name, shorter.name
                )
            }
            CtxError::IllegalProjection => "illegal projection".to_string(),
        }
    }
}

impl From<CtxError> for SubTyError {
    fn from(err: CtxError) -> Self {
        SubTyError::CtxError(err)
    }
}

#[must_use]
#[derive(Debug)]
pub enum BorrowingError {
    Conflict {
        checked: PlaceExpr,
        existing: PlaceExpr,
    },
    ConflictPrevAccess {
        checked: PlaceExpr,
        existing: PlaceExpr,
    },
    CtxError(CtxError),
    // "Trying to use place expression with {} capability while it refers to a \
    //     loan with {} capability.",
    // checked_own, ref_own
    ConflictingOwnership,
    ConflictingAccess,
    // The borrowing place is not in the reborrow list
    BorrowNotInReborrowList(Place),
    TemporaryConflictingBorrow(String),
    WrongDevice(BaseExec, BaseExec),
    MultipleDistribs,
    CannotNarrow,
    DivergingExec,
    TyError(Box<TyError>),
    NatEvalError(NatEvalError),
}

impl BorrowingError {
    fn span(&self) -> Option<Span> {
        match self {
            BorrowingError::Conflict { checked, existing }
            | BorrowingError::ConflictPrevAccess { checked, existing } => {
                checked.span.or(existing.span)
            }
            BorrowingError::CtxError(error) => error.span(),
            BorrowingError::TyError(error) => error.span(),
            _ => None,
        }
    }

    fn message(&self) -> String {
        match self {
            BorrowingError::Conflict { .. } | BorrowingError::ConflictPrevAccess { .. } => {
                "borrow conflicts with an existing access".to_string()
            }
            BorrowingError::CtxError(error) => error.message(),
            BorrowingError::ConflictingOwnership => {
                "borrow conflicts with reference ownership".to_string()
            }
            BorrowingError::ConflictingAccess => {
                "borrow conflicts with an existing access".to_string()
            }
            BorrowingError::BorrowNotInReborrowList(_) => {
                "place is not available for reborrowing".to_string()
            }
            BorrowingError::TemporaryConflictingBorrow(name) => {
                format!("temporary borrow conflicts with provenance `{name}`")
            }
            BorrowingError::WrongDevice(_, _) => {
                "borrow crosses incompatible execution locations".to_string()
            }
            BorrowingError::MultipleDistribs => "borrow has multiple distributions".to_string(),
            BorrowingError::CannotNarrow => "borrow cannot be narrowed".to_string(),
            BorrowingError::DivergingExec => "borrow crosses diverging executions".to_string(),
            BorrowingError::TyError(error) => error.message(),
            BorrowingError::NatEvalError(_) => {
                "could not evaluate a natural number in the borrow".to_string()
            }
        }
    }
}

impl From<TyError> for BorrowingError {
    fn from(err: TyError) -> Self {
        BorrowingError::TyError(Box::new(err))
    }
}
impl From<CtxError> for BorrowingError {
    fn from(err: CtxError) -> Self {
        BorrowingError::CtxError(err)
    }
}
impl From<NatEvalError> for BorrowingError {
    fn from(err: NatEvalError) -> Self {
        BorrowingError::NatEvalError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::internal::Place;
    use crate::ast::{DataTyKind, ExprKind, Nat, NatCtx, PlaceExprKind, ScalarTy};

    fn span() -> Span {
        Span::new(7, 9)
    }

    fn ident() -> Ident {
        Ident::with_span("x", span())
    }

    fn place() -> PlaceExpr {
        PlaceExpr::with_span(PlaceExprKind::Ident(ident()), span())
    }

    fn unspanned_place() -> PlaceExpr {
        PlaceExpr::new(PlaceExprKind::Ident(Ident::new("x")))
    }

    fn data_ty() -> DataTy {
        DataTy::with_span(DataTyKind::Scalar(ScalarTy::I32), span())
    }

    fn ty() -> Ty {
        Ty::with_span(TyKind::Data(Box::new(data_ty())), span())
    }

    fn nat_error() -> NatEvalError {
        Nat::GridIdx.eval(&NatCtx::new()).unwrap_err()
    }

    fn borrow(error: BorrowingError) -> TyError {
        TyError::ConflictingBorrow(Box::new(place()), Ownership::Uniq, error)
    }

    fn nested_borrow(error: BorrowingError) -> TyError {
        TyError::ConflictingBorrow(Box::new(unspanned_place()), Ownership::Uniq, error)
    }

    macro_rules! cases {
        ($count:expr; $($error:expr => $located:expr),+ $(,)?) => {
            let cases = [$(($error, $located)),+];
            assert_eq!(cases.len(), $count);
            for (error, located) in cases {
                let actual = error.span().map(|span| (span.begin, span.end));
                assert_eq!(actual, located.then_some((7, 9)), "{error:?}");
            }
        };
    }

    #[test]
    fn top_level_error_spans() {
        cases!(30;
            TyError::MultiError(vec![TyError::AssignToConst(place())]) => true,
            TyError::MutabilityNotAllowed(ty()) => true,
            TyError::CtxError(CtxError::IdentNotFound(ident())) => true,
            TyError::SubTyError(SubTyError::MemoryKindsNoMatch) => false,
            TyError::MismatchedDataTypes(data_ty(), data_ty(), Expr::with_span(ExprKind::Hole, span())) => true,
            borrow(BorrowingError::ConflictingAccess) => true,
            TyError::PrvValueAlreadyInUse("r".into()) => false,
            TyError::ReferenceToIncompatibleType => false,
            TyError::ReferenceToWrongOwnership => false,
            TyError::ReferenceToDeadTy => false,
            TyError::AssignToConst(place()) => true,
            TyError::AssignToView => false,
            TyError::SplittingNonViewArray => false,
            TyError::ExpectedTupleType(TyKind::Data(Box::new(data_ty())), place()) => true,
            TyError::ConstBorrow(place()) => true,
            TyError::CannotUniquelyDereferenceShared(Some(span())) => true,
            TyError::BorrowingDeadView => false,
            TyError::IllegalExec => false,
            TyError::DeadTy => false,
            TyError::MissingParallelCollectionIdent => false,
            TyError::CouldNotInferProvenance => false,
            TyError::PatternAndTypeDoNotMatch => false,
            TyError::UnexpectedType => false,
            TyError::IllegalDimension => false,
            TyError::UnifyError(UnifyError::CannotUnifyTy(ty(), ty())) => true,
            TyError::MissingMain => false,
            TyError::NatEvalError(nat_error()) => false,
            TyError::CannotInferGenericArg(ident()) => true,
            TyError::UnsafeRequired => false,
            TyError::String("failure".into()) => false,
        );
    }

    #[test]
    fn context_error_spans() {
        cases!(6;
            TyError::CtxError(CtxError::IdentNotFound(ident())) => true,
            TyError::CtxError(CtxError::KindedIdentNotFound(ident())) => true,
            TyError::CtxError(CtxError::PrvValueNotFound("r".into())) => false,
            TyError::CtxError(CtxError::PrvIdentNotFound(ident())) => true,
            TyError::CtxError(CtxError::OutlRelNotDefined(ident(), ident())) => true,
            TyError::CtxError(CtxError::IllegalProjection) => false,
        );
    }

    #[test]
    fn subtyping_error_spans() {
        cases!(7;
            TyError::SubTyError(SubTyError::CtxError(CtxError::IdentNotFound(ident()))) => true,
            TyError::SubTyError(SubTyError::NotOutliving("a".into(), "b".into())) => false,
            TyError::SubTyError(SubTyError::PrvNotUsedInBorrow("r".into())) => false,
            TyError::SubTyError(SubTyError::MemoryKindsNoMatch) => false,
            TyError::SubTyError(SubTyError::OwnershipNoMatch) => false,
            TyError::SubTyError(SubTyError::SizesNoMatch) => false,
            TyError::SubTyError(SubTyError::Dummy) => false,
        );
    }

    #[test]
    fn unification_error_spans() {
        cases!(5;
            TyError::UnifyError(UnifyError::CannotUnify) => false,
            TyError::UnifyError(UnifyError::CannotUnifyTy(ty(), ty())) => true,
            TyError::UnifyError(UnifyError::CannotUnifyDataTy(data_ty(), data_ty())) => true,
            TyError::UnifyError(UnifyError::InfiniteType) => false,
            TyError::UnifyError(UnifyError::SubTyError(SubTyError::CtxError(CtxError::IdentNotFound(ident())))) => true,
        );
    }

    #[test]
    fn borrowing_error_spans() {
        cases!(13;
            nested_borrow(BorrowingError::Conflict { checked: place(), existing: place() }) => true,
            nested_borrow(BorrowingError::ConflictPrevAccess { checked: place(), existing: place() }) => true,
            nested_borrow(BorrowingError::CtxError(CtxError::IdentNotFound(ident()))) => true,
            borrow(BorrowingError::ConflictingOwnership) => true,
            borrow(BorrowingError::ConflictingAccess) => true,
            borrow(BorrowingError::BorrowNotInReborrowList(Place::new(ident(), vec![]))) => true,
            borrow(BorrowingError::TemporaryConflictingBorrow("r".into())) => true,
            borrow(BorrowingError::WrongDevice(BaseExec::CpuThread, BaseExec::CpuThread)) => true,
            borrow(BorrowingError::MultipleDistribs) => true,
            borrow(BorrowingError::CannotNarrow) => true,
            borrow(BorrowingError::DivergingExec) => true,
            nested_borrow(BorrowingError::TyError(Box::new(TyError::AssignToConst(place())))) => true,
            borrow(BorrowingError::NatEvalError(nat_error())) => true,
        );
    }

    #[test]
    fn optional_and_composite_spans() {
        assert!(TyError::CannotUniquelyDereferenceShared(None)
            .span()
            .is_none());
        assert!(TyError::MultiError(vec![]).span().is_none());
        let error = TyError::MultiError(vec![
            TyError::String("first".into()),
            TyError::AssignToConst(place()),
        ]);
        assert_eq!(
            error.span().map(|span| (span.begin, span.end)),
            Some((7, 9))
        );
    }
}
