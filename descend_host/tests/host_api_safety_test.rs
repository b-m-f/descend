#[test]
fn compile_safety_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/host_api_safety/reject_dynamic_vec.rs");
    t.compile_fail("tests/host_api_safety/reject_dynamic_slice.rs");
}
