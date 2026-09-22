#[test]
fn compile_safety_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/host_api_safety/reject_dynamic_vec.rs");
    t.compile_fail("tests/host_api_safety/reject_dynamic_slice.rs");
    t.compile_fail("tests/host_api_safety/require_unsafe_launch.rs");
    t.compile_fail("tests/host_api_safety/require_unsafe_macro.rs");
    t.compile_fail("tests/host_api_safety/borrowed_host_is_exclusive.rs");
    t.compile_fail("tests/host_api_safety/kernel_arg_outlives_buffer.rs");
    t.compile_fail("tests/host_api_safety/mut_arg_is_exclusive.rs");
    t.compile_fail("tests/host_api_safety/scalar_requires_descend_type.rs");
    t.compile_fail("tests/host_api_safety/scalar_arg_outlives_value.rs");
    t.compile_fail("tests/host_api_safety/reject_oversized_array.rs");
}
