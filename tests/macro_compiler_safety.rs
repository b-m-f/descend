#[test]
fn compile_safety_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/macro_compiler_safety/bad_size_test.rs");
    t.compile_fail("tests/macro_compiler_safety/bad_arithmetic_size.rs");
    t.compile_fail("tests/macro_compiler_safety/bad_mutability.rs");
    t.compile_fail("tests/macro_compiler_safety/wrong_arg_count.rs");
    t.compile_fail("tests/macro_compiler_safety/wrong_element_type.rs");
    t.compile_fail("tests/macro_compiler_safety/reject_shared_memory_param.rs");
}
