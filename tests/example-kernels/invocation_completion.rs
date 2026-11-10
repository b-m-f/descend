use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};
use std::time::{Duration, Instant};

const ROW_COUNT: usize = 2048;
const REDUCTION_LENGTH: usize = 2048;
const COLUMN_COUNT: usize = 2048;

fn boxed_matrix<const ELEMENT_COUNT: usize>(value: i32) -> Box<[i32; ELEMENT_COUNT]> {
    vec![value; ELEMENT_COUNT].into_boxed_slice().try_into().unwrap()
}

descend! {
    fn gpu_long_running_matmul<block_edge: nat, row_count: nat, reduction_length: nat, column_count: nat, r: prv>(
        left_matrix: &r shrd gpu.global [i32; row_count * reduction_length],
        right_matrix: &r shrd gpu.global [i32; reduction_length * column_count],
        product_matrix: &r uniq gpu.global [i32; row_count * column_count]
    ) -[grid: gpu.grid<XY<column_count/block_edge, row_count/block_edge>, XY<block_edge, block_edge>>]-> () {
        sched(Y) block_row in grid {
            sched(X) block_column in block_row {
                sched(Y) thread_row in block_column {
                    sched(X) thread_column in thread_row {
                        let left_row = &shrd (*left_matrix).to_view.grp::<reduction_length>.grp::<block_edge>[[block_row]][[thread_row]];
                        let right_column = &shrd (*right_matrix).to_view.grp::<column_count>.transp.grp::<block_edge>[[block_column]][[thread_column]];
                        let mut product_element = &uniq (*product_matrix).to_view.grp::<column_count>.grp::<block_edge>[[block_row]].transp.grp::<block_edge>[[block_column]].transp[[thread_row]][[thread_column]];

                        let mut sum = 0;
                        for index in 0..reduction_length {
                            sum = sum + (*left_row)[index] * (*right_column)[index]
                        };
                        *product_element = sum
                    }
                }
            }
        }
    }
}

#[test]
fn the_invocation_function_returns_only_after_the_kernel_has_completed() {
    let long_running_matmul = descend_fn!(
        gpu_long_running_matmul<<<(128, 128), (16, 16)>>>,
        <block_edge: 16, row_count: 2048, reduction_length: 2048, column_count: 2048>
    );

    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();

    let left_matrix = boxed_matrix::<{ ROW_COUNT * REDUCTION_LENGTH }>(3i32);
    let right_matrix = boxed_matrix::<{ REDUCTION_LENGTH * COLUMN_COUNT }>(7i32);
    let product_matrix = boxed_matrix::<{ ROW_COUNT * COLUMN_COUNT }>(0i32);

    let left_gpu = GpuBuffer::from_host(&left_matrix, &stream).unwrap();
    let right_gpu = GpuBuffer::from_host(&right_matrix, &stream).unwrap();
    let mut product_gpu = GpuBuffer::from_host(&product_matrix, &stream).unwrap();

    long_running_matmul(&stream, &left_gpu, &right_gpu, &mut product_gpu).unwrap();
    context.synchronize().unwrap();

    let measured_invocation_started_at = Instant::now();
    long_running_matmul(&stream, &left_gpu, &right_gpu, &mut product_gpu).unwrap();
    let measured_invocation_duration = measured_invocation_started_at.elapsed();

    let residual_synchronize_started_at = Instant::now();
    context.synchronize().unwrap();
    let residual_synchronize_duration = residual_synchronize_started_at.elapsed();

    println!(
        "the measured invocation took {measured_invocation_duration:?}, \
         the synchronize after it took {residual_synchronize_duration:?}"
    );

    let product_on_host = product_gpu.to_host(&stream).unwrap();
    let expected_element = 3i32 * 7i32 * REDUCTION_LENGTH as i32;
    assert!(
        product_on_host
            .iter()
            .all(|element| *element == expected_element),
        "the invocation returned without the kernel's result being in the buffer"
    );

    assert!(
        residual_synchronize_duration * 4 <= measured_invocation_duration,
        "the invocation returned with device work still pending: the synchronize after it \
         took {residual_synchronize_duration:?}, the invocation itself \
         {measured_invocation_duration:?}"
    );
    assert!(
        measured_invocation_duration >= Duration::from_millis(10),
        "the kernel must occupy the device long enough for the ordering to be observable, \
         the measured invocation took {measured_invocation_duration:?}"
    );
}

#[test]
fn releasing_the_argument_buffers_after_the_invocation_records_no_device_error() {
    let long_running_matmul = descend_fn!(
        gpu_long_running_matmul<<<(128, 128), (16, 16)>>>,
        <block_edge: 16, row_count: 2048, reduction_length: 2048, column_count: 2048>
    );

    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();

    let left_matrix = boxed_matrix::<{ ROW_COUNT * REDUCTION_LENGTH }>(3i32);
    let right_matrix = boxed_matrix::<{ REDUCTION_LENGTH * COLUMN_COUNT }>(7i32);
    let product_matrix = boxed_matrix::<{ ROW_COUNT * COLUMN_COUNT }>(0i32);

    let left_gpu = GpuBuffer::from_host(&left_matrix, &stream).unwrap();
    let right_gpu = GpuBuffer::from_host(&right_matrix, &stream).unwrap();
    let mut product_gpu = GpuBuffer::from_host(&product_matrix, &stream).unwrap();

    long_running_matmul(&stream, &left_gpu, &right_gpu, &mut product_gpu).unwrap();

    drop(product_gpu);
    drop(right_gpu);
    drop(left_gpu);
    drop(stream);

    context.synchronize().unwrap();
    context.check_deferred_errors().expect(
        "releasing the argument buffers and the stream after the invocation returned \
         must not record a device error",
    );
}
