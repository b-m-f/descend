#[cfg(target_os = "linux")]
mod linux_memory_tests {
    use descend_host::{DescendContext, GpuBuffer};
    use std::fs;
    use std::time::Instant;

    fn get_rss_bytes() -> usize {
        let statm = fs::read_to_string("/proc/self/statm").unwrap();
        let parts: Vec<&str> = statm.split_whitespace().collect();
        let rss_pages: usize = parts[1].parse().unwrap();
        let page_size = 4096;
        rss_pages * page_size
    }

    #[test]
    fn test_from_box_sync_memory_reclaim() {
        let ctx = DescendContext::new(0);
        let rss_baseline = get_rss_bytes();
        
        // Allocate 100MB
        const N: usize = 25_000_000;
        let mut data_vec = vec![0i32; N].into_boxed_slice();
        for i in 0..N { data_vec[i] = 1; }
        let data: Box<[i32; N]> = data_vec.try_into().unwrap();
        
        let rss_allocated = get_rss_bytes();
        assert!(rss_allocated >= rss_baseline + 100_000_000, "Memory should increase after allocation");

        let start = Instant::now();
        let buf = GpuBuffer::from_box(&ctx, data);
        let duration = start.elapsed();
        
        let rss_after_from_box = get_rss_bytes();
        // Since from_box is sync and consumes the Box, it drops it before returning.
        assert!(rss_after_from_box < rss_allocated - 50_000_000, "Memory should be reclaimed immediately by from_box");
        
        println!("Sync from_box took: {:?}", duration);

        let fetched = buf.to_cpu();
        assert_eq!(fetched.len(), N);
        assert_eq!(fetched[0], 1, "Data at index 0 corrupted or lost");
        assert_eq!(fetched[N - 1], 1, "Data at end of buffer corrupted or lost");
    }

    #[test]
    fn test_from_slice_sync_memory_retained() {
        let ctx = DescendContext::new(0);
        let rss_baseline = get_rss_bytes();
        
        // Allocate 100MB
        const N: usize = 25_000_000;
        let mut data_vec = vec![0i32; N].into_boxed_slice();
        for i in 0..N { data_vec[i] = 1; }
        let data: Box<[i32; N]> = data_vec.try_into().unwrap();
        
        let rss_allocated = get_rss_bytes();
        assert!(rss_allocated >= rss_baseline + 100_000_000, "Memory should increase after allocation");

        let start = Instant::now();
        let buf = GpuBuffer::from_slice(&ctx, &data);
        let duration = start.elapsed();
        
        println!("Sync from_slice took: {:?}", duration);
        
        let rss_after_from_slice = get_rss_bytes();
        // The memory is retained because we passed a reference
        assert!(rss_after_from_slice >= rss_baseline + 100_000_000, "Memory must remain allocated while consumer holds array");

        // Manually drop the host memory
        drop(data);
        
        let rss_final = get_rss_bytes();
        assert!(rss_final < rss_allocated - 50_000_000, "Memory should be reclaimed after consumer drops array");

        let fetched = buf.to_cpu();
        assert_eq!(fetched.len(), N);
        assert_eq!(fetched[0], 1, "Data at index 0 corrupted or lost");
        assert_eq!(fetched[N - 1], 1, "Data at end of buffer corrupted or lost");
    }
}
