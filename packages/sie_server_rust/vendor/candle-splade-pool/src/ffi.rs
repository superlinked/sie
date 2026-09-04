use core::ffi::{c_int, c_void};

extern "C" {
    pub(crate) fn splade_segmented_max_log1p_f16(
        input: *const c_void,
        offsets: *const c_void,
        output: *mut c_void,
        total_tokens: i64,
        vocab_size: c_int,
        batch_size: c_int,
        stream: *mut c_void,
    ) -> c_int;
}
