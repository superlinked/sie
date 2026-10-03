use core::ffi::{c_int, c_void};

extern "C" {
    pub(crate) fn run_ln(
        x: *const c_void,
        residual: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        dst_add: *mut c_void,
        dst: *mut c_void,
        mu: *mut c_void,
        rsigma: *mut c_void,

        epsilon: f32,

        hidden_size_rounded: u32,
        rows: u32,
        cols: u32,
        multi_processor_count: i32,

        wtype: u32,
        itype: u32,
        rtype: u32,
        otype: u32,
        ctype: u32,

        is_rms_norm: c_int,
        stream: *mut c_void,
    ) -> c_int;
}
