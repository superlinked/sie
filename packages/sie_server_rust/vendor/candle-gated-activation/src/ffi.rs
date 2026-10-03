use core::ffi::{c_int, c_long, c_void};

extern "C" {
    pub(crate) fn init_gelu_erf_bf16_lut(lut: *mut c_void, stream: *mut c_void) -> c_int;

    pub(crate) fn gelu_gate(
        input: *const c_void,
        output: *mut c_void,
        rows: c_long,
        intermediate_size: c_int,
        dtype: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub(crate) fn gelu_erf_gate(
        input: *const c_void,
        output: *mut c_void,
        rows: c_long,
        intermediate_size: c_int,
        dtype: u32,
        stream: *mut c_void,
    ) -> c_int;

    pub(crate) fn gelu_erf_gate_bf16_lut(
        input: *const c_void,
        output: *mut c_void,
        lut: *const c_void,
        rows: c_long,
        intermediate_size: c_int,
        stream: *mut c_void,
    ) -> c_int;
}
