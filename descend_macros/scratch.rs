pub struct GpuUniq<T, const N: usize>(std::marker::PhantomData<T>);
fn main() {
    let mut a: GpuUniq<i32, 1024> = GpuUniq(std::marker::PhantomData);
    let mut a_ref = &mut a;
    let _check: &mut GpuUniq<_, 1024> = a_ref;
    
    // This should fail:
    // let _check2: &mut GpuUniq<_, 2048> = a_ref;
}
