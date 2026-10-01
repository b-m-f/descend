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
}
