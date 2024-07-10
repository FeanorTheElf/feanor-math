#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#[repr(C)]
#[derive(Clone, Copy)]
pub struct __mpz_struct {
    _mp_alloc: libc::c_int,
    _mp_size: libc::c_int,
    _mp_d: *mut libc::c_void
}

pub const UNINIT_MPZ: __mpz_struct = __mpz_struct {
    _mp_alloc: 0,
    _mp_size: 0,
    _mp_d: std::ptr::null_mut()
};

pub type mpz_srcptr = *const __mpz_struct;
pub type mpz_ptr = *mut __mpz_struct;

#[cfg(windows)]
pub type mpir_si = libc::c_longlong;
#[cfg(not(windows))]
pub type mpir_si = libc::c_long;

#[cfg(windows)]
pub type mpir_ui = libc::c_ulonglong;
#[cfg(not(windows))]
pub type mpir_ui = libc::c_ulong;

///
/// Returns true if the value is negative, and false if it is positive.
/// For 0, the result is arbitrary.
///
pub unsafe fn mpz_is_neg(val: mpz_srcptr) -> bool {
    // sadly, the function mpz_sgn() is only a macro in mpir (with more or
    // less this implementation), and so we have to break into the internals
    // of mpir here.
    // In particular, they indicate a negative value by having size to be
    // negative (its absolute value is still the "real" size)
    (*val)._mp_size < 0
}

#[link(name = "mpir", kind = "static")]
extern "C" {
    //
    // It is a very hidden, but important fact that we are allowed
    // to pass the same pointer as multiple arguments, even in and
    // output parameters. See doc 3.0.0, MPIR Basics > Parameter Conventions,
    // directly after the source code example (only a half-sentence).
    // 

    pub fn __gmpz_init(val: mpz_ptr);
    pub fn __gmpz_clear(val: mpz_ptr);
    pub fn __gmpz_add(dst: mpz_ptr, fst: mpz_srcptr, snd: mpz_srcptr);
    pub fn __gmpz_add_ui(dst: mpz_ptr, fst: mpz_srcptr, snd: mpir_ui);
    pub fn __gmpz_sub(dst: mpz_ptr, fst: mpz_srcptr, snd: mpz_srcptr);
    pub fn __gmpz_sub_ui(dst: mpz_ptr, fst: mpz_srcptr, snd: mpir_ui);
    pub fn __gmpz_ui_sub(dst: mpz_ptr, fst: mpir_ui, snd: mpz_srcptr);
    pub fn __gmpz_addmul(dst: mpz_ptr, fst: mpz_srcptr, snd: mpz_srcptr);
    pub fn __gmpz_submul(dst: mpz_ptr, fst: mpz_srcptr, snd: mpz_srcptr);
    pub fn __gmpz_mul_2exp(dst: mpz_ptr, fst: mpz_srcptr, snd: mpir_ui);
    pub fn __gmpz_mul(dst: mpz_ptr, fst: mpz_srcptr, snd: mpz_srcptr);
    pub fn __gmpz_mul_ui(dst: mpz_ptr, fst: mpz_srcptr, snd: mpir_ui);
    pub fn __gmpz_neg(dst: mpz_ptr, fst: mpz_srcptr);
    pub fn __gmpz_set(dst: mpz_ptr, fst: mpz_srcptr);
    pub fn __gmpz_set_d(dst: mpz_ptr, val: libc::c_double);
    pub fn __gmpz_abs(dst: mpz_ptr, fst: mpz_srcptr);
    pub fn __gmpz_tdiv_q(q: mpz_ptr, n: mpz_srcptr, d: mpz_srcptr);
    pub fn __gmpz_tdiv_r(r: mpz_ptr, n: mpz_srcptr, d: mpz_srcptr);
    pub fn __gmpz_tdiv_qr(q: mpz_ptr, r: mpz_ptr, n: mpz_srcptr, d: mpz_srcptr);
    pub fn __gmpz_set_ui(dst: mpz_ptr, val: mpir_ui);
    pub fn __gmpz_set_si(dst: mpz_ptr, val: mpir_si);
    pub fn __gmpz_tdiv_q_2exp(dst: mpz_ptr, val: mpz_srcptr, pow: mpir_ui);
    /// returns the least significant bits if the value is to large
    pub fn __gmpz_get_si(val: mpz_srcptr) -> mpir_si;
    /// returns only the least significant bits; works with abs(val)
    pub fn __gmpz_get_ui(val: mpz_srcptr) -> mpir_ui;
    pub fn __gmpz_get_d(val: mpz_srcptr) -> libc::c_double;
    pub fn __gmpz_cmp(lhs: mpz_srcptr, rhs: mpz_srcptr) -> libc::c_int;
    pub fn __gmpz_cmp_si(lhs: mpz_srcptr, rhs: mpir_si) -> libc::c_int;
    pub fn __gmpz_cmpabs(lhs: mpz_srcptr, rhs: mpz_srcptr) -> libc::c_int;
    // this may be one too large, unless base is a power of two
    pub fn __gmpz_sizeinbase(val: mpz_srcptr, base: libc::c_int) -> libc::size_t;
    pub fn __gmpz_tstbit(val: mpz_srcptr, bit_index: mpir_ui) -> libc::c_int;
    pub fn __gmpz_fdiv_q(dst: mpz_ptr, lhs: mpz_srcptr, rhs: mpz_srcptr);
    pub fn __gmpz_nthroot(dst: mpz_ptr, val: mpz_srcptr, n: mpir_ui);
    pub fn __gmpz_export(dst: *mut libc::c_void, countp: *mut libc::size_t, order: libc::c_int, size: libc::size_t, endian: libc::c_int, nails: libc::size_t, data: mpz_srcptr) -> *mut libc::c_void;
    pub fn __gmpz_import(dst: mpz_ptr, count: libc::size_t, order: libc::c_int, size: libc::size_t, endian: libc::c_int, nails: libc::size_t, data: *const libc::c_void);
    pub fn __gmpz_scan1(val: mpz_srcptr, starting_bit: mpir_ui) -> mpir_ui;
    pub fn __gmpz_scan0(val: mpz_srcptr, starting_bit: mpir_ui) -> mpir_ui;
}

#[test]
pub fn test__gmpz_neg() {
    unsafe {
        let mut integer = UNINIT_MPZ;
        __gmpz_init(&mut integer as mpz_ptr);
        __gmpz_set_si(&mut integer as mpz_ptr, 1);
        __gmpz_neg(&mut integer as mpz_ptr, &integer as mpz_srcptr);
        let result = __gmpz_get_si(&integer as mpz_srcptr);
        assert_eq!(-1, result);
        __gmpz_clear(&mut integer as mpz_ptr);
    }
}

#[test]
pub fn test___gmpz_add() {
    unsafe {
        let mut integer = UNINIT_MPZ;
        __gmpz_init(&mut integer as mpz_ptr);
        __gmpz_set_si(&mut integer as mpz_ptr, 14);
        __gmpz_add(&mut integer as mpz_ptr, &integer as mpz_srcptr, &integer as mpz_srcptr);
        let result = __gmpz_get_si(&integer as mpz_srcptr);
        assert_eq!(14 + 14, result);
        __gmpz_clear(&mut integer as mpz_ptr);
    }
}

#[test]
pub fn test___gmpz_mul() {
    unsafe {
        let mut integer = UNINIT_MPZ;
        __gmpz_init(&mut integer as mpz_ptr);
        __gmpz_set_si(&mut integer as mpz_ptr, 14);
        __gmpz_mul(&mut integer as mpz_ptr, &integer as mpz_srcptr, &integer as mpz_srcptr);
        let result = __gmpz_get_si(&integer as mpz_srcptr);
        assert_eq!(14 * 14, result);
        __gmpz_clear(&mut integer as mpz_ptr);
    }
}

#[test]
pub fn test___gmpz_sub() {
    unsafe {
        let mut a = UNINIT_MPZ;
        let mut b = UNINIT_MPZ;
        __gmpz_init(&mut a as mpz_ptr);
        __gmpz_init(&mut b as mpz_ptr);
        __gmpz_set_si(&mut a as mpz_ptr, 14);
        __gmpz_set_si(&mut b as mpz_ptr, 13);
        __gmpz_sub(&mut a as mpz_ptr, &a as mpz_srcptr, &b as mpz_srcptr);
        let result = __gmpz_get_si(&a as mpz_srcptr);
        assert_eq!(14 -13, result);
        __gmpz_clear(&mut a as mpz_ptr);
        __gmpz_clear(&mut b as mpz_ptr);
    }
}

#[test]
pub fn test___gmpz_addmul() {
    unsafe {
        let mut integer = UNINIT_MPZ;
        __gmpz_init(&mut integer as mpz_ptr);
        __gmpz_set_si(&mut integer as mpz_ptr, 14);
        __gmpz_addmul(&mut integer as mpz_ptr, &integer as mpz_srcptr, &integer as mpz_srcptr);
        let result = __gmpz_get_si(&integer as mpz_srcptr);
        assert_eq!(14 + 14 * 14, result);
        __gmpz_clear(&mut integer as mpz_ptr);
    }
}

#[test]
pub fn test___gmpz_mul_2exp() {
    unsafe {
        let mut integer = UNINIT_MPZ;
        __gmpz_init(&mut integer as mpz_ptr);
        __gmpz_set_si(&mut integer as mpz_ptr, 14);
        __gmpz_mul_2exp(&mut integer as mpz_ptr, &integer as mpz_srcptr, 4);
        let result = __gmpz_get_si(&integer as mpz_srcptr);
        assert_eq!(14 << 4, result);
        __gmpz_clear(&mut integer as mpz_ptr);
    }
}

#[test]
pub fn test___gmpz_tdiv_qr() {
    unsafe {
        let mut a = UNINIT_MPZ;
        let mut b = UNINIT_MPZ;
        __gmpz_init(&mut a as mpz_ptr);
        __gmpz_init(&mut b as mpz_ptr);
        __gmpz_set_si(&mut a as mpz_ptr, 14);
        __gmpz_set_si(&mut b as mpz_ptr, 5);
        __gmpz_tdiv_qr(&mut a as mpz_ptr, &mut b as mpz_ptr, &a as mpz_srcptr, &b as mpz_srcptr);
        let a_res = __gmpz_get_si(&a as mpz_srcptr);
        let b_res = __gmpz_get_si(&b as mpz_srcptr);
        assert_eq!(2, a_res);
        assert_eq!(4, b_res);
        __gmpz_clear(&mut a as mpz_ptr);
        __gmpz_clear(&mut b as mpz_ptr);
    }
}