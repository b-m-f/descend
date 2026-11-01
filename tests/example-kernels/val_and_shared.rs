use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

descend! {
    fn gpu_scale_through_tile<r1: prv, r2: prv, r3: prv>(
        source: &r1 shrd gpu.global [i32; 64],
        factor: i32,
        destination: &r2 uniq gpu.global [i32; 64],
        [grid.forall(X)] tile: &r3 uniq gpu.shared [i32; 64]
    ) -[grid: gpu.grid<X<1>, X<64>>]-> () {
        let source_view = &shrd (*source).to_view.grp::<64>;
        let destination_view = &uniq (*destination).to_view.grp::<64>;
        sched block in grid {
            let source_block = &shrd (*source_view)[[block]];
            let destination_block = &uniq (*destination_view)[[block]];
            let tile_view = &uniq (*tile).to_view;
            sched thread in block {
                let source_element = &shrd (*source_block)[[thread]];
                let tile_element = &uniq (*tile_view)[[thread]];
                *tile_element = *source_element * factor
            };
            sync(block);
            sched thread in block {
                let tile_element = &shrd (*tile_view)[[thread]];
                let destination_element = &uniq (*destination_block)[[thread]];
                *destination_element = *tile_element
            }
        }
    }
}

#[test]
fn test_a_value_a_buffer_and_a_shared_tile_share_one_signature() {
    let gpu_scale_through_tile = descend_fn!(gpu_scale_through_tile);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let source_gpu = GpuBuffer::from_host(&[3i32; 64], &stream).unwrap();
    let mut destination_gpu = GpuBuffer::from_host(&[0i32; 64], &stream).unwrap();

    gpu_scale_through_tile(&stream, &source_gpu, 7, &mut destination_gpu)
        .unwrap();

    assert_eq!(
        *destination_gpu.to_host(&stream).unwrap(),
        [21i32; 64]
    );
}
