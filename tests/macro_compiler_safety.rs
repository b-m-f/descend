#[test]
fn compile_safety_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/macro_compiler_safety/bad_size_test.rs");
    t.compile_fail("tests/macro_compiler_safety/bad_arithmetic_size.rs");
    t.compile_fail("tests/macro_compiler_safety/bad_mutability.rs");
}
