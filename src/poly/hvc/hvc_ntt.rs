use super::hvc_ntt_param::{INV_NTT_TABLE, NTT_TABLE};
use crate::normalize;
use crate::{impl_ntt_poly, impl_poly, HVCPoly, HVC_ONE_OVER_N};
use crate::{HVC_MODULUS, N};

#[derive(Debug, Clone, Copy)]
// HVC polynomials in NTT encoding
pub struct HVCNTTPoly {
    pub(crate) coeffs: [i32; N],
}

impl_poly!(HVCNTTPoly, HVC_MODULUS, N);
impl_ntt_poly!(HVCPoly, HVCNTTPoly, HVC_MODULUS, N, HVC_ONE_OVER_N);

// =============================================================
// Fast NTT path (feature = "fast-ntt"): scalar Barrett + AVX2 dispatch.
// Bit-exact (mod q) with the reference `% q` implementation.
// =============================================================

#[cfg(feature = "fast-ntt")]
const Q: i32 = HVC_MODULUS;
#[cfg(feature = "fast-ntt")]
const Q64: i64 = HVC_MODULUS as i64;
// Forward butterfly multiplies values in [0, q) by twiddle in [0, q): product
// in [0, q^2) < 2^36. Inverse butterfly pre-canonicalises the subtract before
// multiplying, so K=36 covers both. q*m fits in i64 (18 + 18 bits).
#[cfg(feature = "fast-ntt")]
pub(super) const HVC_BARRETT_K: u32 = 36;
#[cfg(feature = "fast-ntt")]
pub(super) const HVC_BARRETT_M: i64 = (1i64 << HVC_BARRETT_K) / Q64;

#[cfg(feature = "fast-ntt")]
#[inline(always)]
pub(super) fn barrett(x: i64) -> i32 {
    // x in [0, 2^36).
    let t = (x.wrapping_mul(HVC_BARRETT_M)) >> HVC_BARRETT_K;
    let r = x - t * Q64;
    let r = if r >= Q64 { r - Q64 } else { r };
    r as i32
}

#[cfg(feature = "fast-ntt")]
#[inline(always)]
pub(super) fn add_mod(a: i32, b: i32) -> i32 {
    let s = a + b;
    if s >= Q { s - Q } else { s }
}

#[cfg(feature = "fast-ntt")]
#[inline(always)]
pub(super) fn sub_mod(a: i32, b: i32) -> i32 {
    let d = a - b;
    if d < 0 { d + Q } else { d }
}

#[cfg(feature = "fast-ntt")]
#[inline(always)]
pub(super) fn canon(x: i32) -> i32 {
    if x < 0 { x + Q } else { x }
}

/// Scalar butterfly stages of the forward NTT, starting at stage `start_l`
/// with `t = 1 << (9 - start_l)`. Inputs must already be in `[0, q)`.
#[cfg(feature = "fast-ntt")]
pub(super) fn ntt_stages_scalar(p: &mut [i32; N], start_l: usize, mut t: usize) {
    for l in start_l..crate::LOG_N {
        let m = 1usize << l;
        let ht = t >> 1;
        let mut i = 0usize;
        let mut j1 = 0usize;
        while i < m {
            let s = canon(NTT_TABLE[m + i]);
            let j2 = j1 + ht;
            let mut j = j1;
            while j < j2 {
                let u = p[j];
                let v = barrett((p[j + ht] as i64) * (s as i64));
                p[j] = add_mod(u, v);
                p[j + ht] = sub_mod(u, v);
                j += 1;
            }
            i += 1;
            j1 += t;
        }
        t = ht;
    }
}

/// Scalar butterfly stages of the inverse NTT, run for `iters` iterations
/// starting from `t = start_t`, `m = start_m`. Inputs must be in `[0, q)`.
/// Returns updated `(t, m)` for resumption.
#[cfg(feature = "fast-ntt")]
pub(super) fn inv_ntt_stages_scalar(
    p: &mut [i32; N],
    iters: usize,
    mut t: usize,
    mut m: usize,
) -> (usize, usize) {
    for _ in 0..iters {
        if m <= 1 {
            break;
        }
        let hm = m >> 1;
        let dt = t << 1;
        let mut i = 0usize;
        let mut j1 = 0usize;
        while i < hm {
            let j2 = j1 + t;
            let s = canon(INV_NTT_TABLE[hm + i]);
            let mut j = j1;
            while j < j2 {
                let u = p[j];
                let v = p[j + t];
                p[j] = add_mod(u, v);
                let d = sub_mod(u, v);
                p[j + t] = barrett((d as i64) * (s as i64));
                j += 1;
            }
            i += 1;
            j1 += dt;
        }
        t = dt;
        m = hm;
    }
    (t, m)
}

#[cfg(feature = "fast-ntt")]
pub(super) fn inv_ntt_final_scale(p: &mut [i32; N]) {
    let inv_n = canon(HVC_ONE_OVER_N);
    for e in p.iter_mut() {
        *e = barrett((*e as i64) * (inv_n as i64));
    }
}

#[cfg(feature = "fast-ntt")]
pub(super) fn ntt_scalar(p: &mut [i32; N]) {
    for e in p.iter_mut() {
        *e = canon(*e);
    }
    ntt_stages_scalar(p, 0, N);
}

#[cfg(feature = "fast-ntt")]
pub(super) fn inv_ntt_scalar(p: &mut [i32; N]) {
    for e in p.iter_mut() {
        *e = canon(*e);
    }
    inv_ntt_stages_scalar(p, crate::LOG_N, 1, N);
    inv_ntt_final_scale(p);
}

#[cfg(feature = "fast-ntt")]
type NttFn = fn(&mut [i32; N]);

#[cfg(feature = "fast-ntt")]
static NTT_DISPATCH: std::sync::OnceLock<NttFn> = std::sync::OnceLock::new();
#[cfg(feature = "fast-ntt")]
static INV_NTT_DISPATCH: std::sync::OnceLock<NttFn> = std::sync::OnceLock::new();

/// The AVX2 i16 Montgomery path is only correct while `q < 2^15` (butterfly
/// intermediates `u ± v_red ∈ [-q, q]` must fit a signed i16). For larger q
/// (e.g. q_hvc = 40961 at N=2048) we fall back to the scalar i32 Barrett path.
#[cfg(feature = "fast-ntt")]
const I16_AVX2_OK: bool = HVC_MODULUS < (1 << 15);

#[cfg(feature = "fast-ntt")]
fn select_ntt() -> NttFn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // q < 2^15 → i16 Montgomery kernel; larger q (e.g. q_hvc = 40961 at
            // N=2048) → the i32 Barrett kernel (`ntt_avx2`). Both beat scalar.
            if I16_AVX2_OK {
                return super::simd::ntt_avx2_i16_dispatch;
            }
            return super::simd::ntt_avx2_dispatch;
        }
    }
    ntt_scalar
}

#[cfg(feature = "fast-ntt")]
fn select_inv_ntt() -> NttFn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            if I16_AVX2_OK {
                return super::simd::inv_ntt_avx2_i16_dispatch;
            }
            return super::simd::inv_ntt_avx2_dispatch;
        }
    }
    inv_ntt_scalar
}

#[cfg(feature = "fast-ntt")]
fn ntt(p: &mut [i32; N]) {
    (NTT_DISPATCH.get_or_init(select_ntt))(p)
}

#[cfg(feature = "fast-ntt")]
fn inv_ntt(p: &mut [i32; N]) {
    (INV_NTT_DISPATCH.get_or_init(select_inv_ntt))(p)
}

#[cfg(test)]
mod test {
    use super::*;
    use super::{inv_ntt, ntt};
    use crate::impl_ntt_poly_tests;
    use crate::poly::Polynomial;
    use crate::{poly::lift, HVCPoly, HVC_MODULUS};
    use ark_std::{end_timer, start_timer};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    impl_ntt_poly_tests!(HVCPoly, HVCNTTPoly, HVC_MODULUS);

    #[cfg(feature = "fast-ntt")]
    #[test]
    fn barrett_matches_rem_sweep() {
        // Exhaustive in step; covers full input range up to 2^36.
        for x in (0i64..(1i64 << 36)).step_by(7919) {
            assert_eq!(barrett(x) as i64, x % Q64, "x={}", x);
        }
    }
}
