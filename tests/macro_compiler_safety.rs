#[test]
fn compile_safety_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/macro_compiler_safety/bad_size_test.rs");
    t.compile_fail("tests/macro_compiler_safety/bad_arithmetic_size.rs");
    t.compile_fail("tests/macro_compiler_safety/bad_mutability.rs");
    t.compile_fail("tests/macro_compiler_safety/wrong_arg_count.rs");
    t.compile_fail("tests/macro_compiler_safety/wrong_element_type.rs");
    t.compile_fail("tests/macro_compiler_safety/shared_param_is_not_an_argument.rs");
    t.compile_fail("tests/macro_compiler_safety/kernel_is_not_launched_by_macro.rs");
    t.compile_fail("tests/macro_compiler_safety/kernel_called_without_an_instantiation.rs");
    t.compile_fail("tests/macro_compiler_safety/size_shadowed_by_binding.rs");
    t.compile_fail("tests/macro_compiler_safety/shared_param_not_last.rs");
    t.compile_fail("tests/macro_compiler_safety/shared_param_unsupported_elem.rs");
    t.compile_fail("tests/macro_compiler_safety/tuple_field_cannot_cross_to_the_host.rs");
    t.compile_fail("tests/macro_compiler_safety/tuple_parameter_cannot_cross_to_the_host.rs");

    t.compile_fail("tests/macro_compiler_safety/descend_declares_no_kernel.rs");
    t.compile_fail("tests/macro_compiler_safety/instantiating_an_undeclared_kernel.rs");
    t.compile_fail("tests/macro_compiler_safety/duplicate_size_binding.rs");
    t.compile_fail("tests/macro_compiler_safety/a_raw_identifier_does_not_panic_the_macro.rs");
    t.compile_fail("tests/macro_compiler_safety/two_kernels_with_the_same_name.rs");
    t.compile_fail("tests/macro_compiler_safety/borrow_conflict_is_reported.rs");
    t.compile_fail("tests/macro_compiler_safety/parse_unexpected_punctuation.rs");
    t.compile_fail("tests/macro_compiler_safety/parse_missing_expression.rs");
    t.compile_fail("tests/macro_compiler_safety/parse_end_of_input.rs");
}
