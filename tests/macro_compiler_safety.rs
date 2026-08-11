#[test]
fn compile_safety_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/macro_compiler_safety/bad_size_test.rs");
}
