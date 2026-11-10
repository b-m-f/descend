extern crate core;

use crate::error::ErrorReported;

pub mod ast;
pub mod codegen;
pub mod error;
pub mod parser;
// FIXME do not allow the large error
//   keep it for now because it complicates pattern matching etc. and is not critical
#[allow(clippy::result_large_err)]
pub mod ty_check;

pub fn compile(file_path: &str) -> Result<String, ErrorReported> {
    let source = parser::SourceCode::from_file(file_path)?;
    let mut compil_unit = parser::parse(&source)?;
    ty_check::ty_check(&mut compil_unit)?;
    Ok(codegen::gen(&compil_unit))
}

/// Expose the C++ header for the procedural macro to use during NVRTC compilation
pub const DESCEND_CUH: &str = include_str!("../cuda-examples/descend.cuh");
