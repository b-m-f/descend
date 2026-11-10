pub const FAULTING_PTX: &str = ".version 7.0
.target sm_52
.address_size 64

.visible .entry fault()
{
\t.reg .u64 \t%rd<2>;
\t.reg .u32 \t%r<2>;
\tmov.u64 \t%rd1, 0;
\tmov.u32 \t%r1, 1;
\tst.global.u32 \t[%rd1], %r1;
\tret;
}
";
