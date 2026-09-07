use descend_host::DescendContext;

#[test]
fn test_multiple_streams() {
    let ctx = DescendContext::new(0);
    let s1 = ctx.new_stream();
    let s2 = ctx.new_stream();
    assert_ne!(s1.inner.cu_stream() as usize, 0, "s1 should not be stream 0");
    assert_ne!(s2.inner.cu_stream() as usize, 0, "s2 should not be stream 0");
    assert_ne!(s1.inner.cu_stream(), s2.inner.cu_stream(), "Streams should be different");
}
