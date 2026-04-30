use super::kahe_ntt_param::{INV_NTT_TABLE, NTT_TABLE};
use crate::normalize;
use crate::{impl_ntt_poly, impl_poly, KahePoly, KAHE_ONE_OVER_N};
use crate::{KAHE_MODULUS, N};

#[derive(Debug, Clone, Copy)]
// KAHE polynomials in NTT encoding
pub struct KaheNTTPoly {
    pub(crate) coeffs: [i32; N as usize],
}

impl_poly!(KaheNTTPoly, KAHE_MODULUS, N);
impl_ntt_poly!(KahePoly, KaheNTTPoly, KAHE_MODULUS, N, KAHE_ONE_OVER_N);

// =============================================================
// Fast NTT path (feature = "fast-ntt"): scalar Barrett + AVX2 dispatch.
// =============================================================

#[cfg(feature = "fast-ntt")]
const Q: i32 = KAHE_MODULUS;
#[cfg(feature = "fast-ntt")]
const Q64: i64 = KAHE_MODULUS as i64;
// Barrett: q ≈ 2^30, product q·s gives < 2^60. K=60, m fits in 31 bits.
#[cfg(feature = "fast-ntt")]
const KAHE_BARRETT_K: u32 = 60;
#[cfg(feature = "fast-ntt")]
const KAHE_BARRETT_M: i64 = (1i64 << KAHE_BARRETT_K) / Q64;

#[cfg(feature = "fast-ntt")]
#[inline(always)]
fn barrett(x: i64) -> i32 {
    // x in [0, 2^60). m·x ≈ 2^91 — use i128 intermediate.
    let t = (((x as i128) * (KAHE_BARRETT_M as i128)) >> KAHE_BARRETT_K) as i64;
    let r = x - t * Q64;
    let r = if r >= Q64 { r - Q64 } else { r };
    let r = if r < 0 { r + Q64 } else { r };
    r as i32
}

#[cfg(feature = "fast-ntt")]
#[inline(always)]
fn add_mod(a: i32, b: i32) -> i32 {
    let s = a + b;
    if s >= Q { s - Q } else { s }
}

#[cfg(feature = "fast-ntt")]
#[inline(always)]
fn sub_mod(a: i32, b: i32) -> i32 {
    let d = a - b;
    if d < 0 { d + Q } else { d }
}

#[cfg(feature = "fast-ntt")]
#[inline(always)]
fn canon(x: i32) -> i32 {
    if x < 0 { x + Q } else { x }
}

/// Scalar butterfly stages of the forward NTT, starting at stage `start_l`
/// with `t = 1 << (9 - start_l)`. Inputs must already be in `[0, q)`.
#[cfg(feature = "fast-ntt")]
pub(super) fn ntt_stages_scalar(p: &mut [i32; N as usize], start_l: usize, mut t: usize) {
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
    p: &mut [i32; N as usize],
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
fn inv_ntt_final_scale(p: &mut [i32; N as usize]) {
    let inv_n = canon(KAHE_ONE_OVER_N);
    for e in p.iter_mut() {
        *e = barrett((*e as i64) * (inv_n as i64));
    }
}

#[cfg(feature = "fast-ntt")]
fn ntt_scalar(p: &mut [i32; N as usize]) {
    for e in p.iter_mut() {
        *e = canon(*e);
    }
    ntt_stages_scalar(p, 0, N as usize);
}

#[cfg(feature = "fast-ntt")]
fn inv_ntt_scalar(p: &mut [i32; N as usize]) {
    for e in p.iter_mut() {
        *e = canon(*e);
    }
    inv_ntt_stages_scalar(p, crate::LOG_N, 1, N as usize);
    inv_ntt_final_scale(p);
}

#[cfg(feature = "fast-ntt")]
type NttFn = fn(&mut [i32; N as usize]);

#[cfg(feature = "fast-ntt")]
static NTT_DISPATCH: std::sync::OnceLock<NttFn> = std::sync::OnceLock::new();
#[cfg(feature = "fast-ntt")]
static INV_NTT_DISPATCH: std::sync::OnceLock<NttFn> = std::sync::OnceLock::new();

#[cfg(feature = "fast-ntt")]
fn select_ntt() -> NttFn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
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
            return super::simd::inv_ntt_avx2_dispatch;
        }
    }
    inv_ntt_scalar
}

#[cfg(feature = "fast-ntt")]
fn ntt(p: &mut [i32; N as usize]) {
    (NTT_DISPATCH.get_or_init(select_ntt))(p)
}

#[cfg(feature = "fast-ntt")]
fn inv_ntt(p: &mut [i32; N as usize]) {
    (INV_NTT_DISPATCH.get_or_init(select_inv_ntt))(p)
}

/// NTT-domain inner product (`Σ a[k] · b[k]`). Re-exported from `simd` so
/// callers see one function that dispatches to AVX2 when available.
pub use super::simd::pointwise_dot;

#[cfg(test)]
mod test {
    use super::*;
    use super::{inv_ntt, ntt};
    use crate::poly::Polynomial;
    use crate::{poly::lift, KahePoly, KAHE_MODULUS};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_conversion() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        for _ in 0..10 {
            let poly = KahePoly::rand_poly(&mut rng);
            let poly_ntt: KaheNTTPoly = (&poly).into();
            let poly_rec: KahePoly = (&poly_ntt).into();
            assert_eq!(poly, poly_rec);
        }
    }

    #[test]
    fn test_arithmetic() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        for _ in 0..10 {
            let a = KahePoly::rand_poly(&mut rng);
            let a_ntt: KaheNTTPoly = (&a).into();
            let b = KahePoly::rand_poly(&mut rng);
            let b_ntt: KaheNTTPoly = (&b).into();

            let c_ntt = a_ntt * b_ntt;
            let c: KahePoly = (&c_ntt).into();
            let c_rec = KahePoly::schoolbook(&a, &b);
            assert_eq!(c, c_rec);

            let d_ntt = a_ntt + b_ntt;
            let d: KahePoly = (&d_ntt).into();
            let d_rec = a + b;
            assert_eq!(d, d_rec);
        }
    }

    #[test]
    fn test_pointwise_dot() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        let a0 = KahePoly::rand_poly(&mut rng);
        let a1 = KahePoly::rand_poly(&mut rng);
        let b0 = KahePoly::rand_poly(&mut rng);
        let b1 = KahePoly::rand_poly(&mut rng);
        let a_ntt: Vec<KaheNTTPoly> = vec![(&a0).into(), (&a1).into()];
        let b_ntt: Vec<KaheNTTPoly> = vec![(&b0).into(), (&b1).into()];
        let dot = pointwise_dot(&a_ntt, &b_ntt);
        let expected = a_ntt[0] * b_ntt[0] + a_ntt[1] * b_ntt[1];
        for (x, y) in dot.coeffs.iter().zip(expected.coeffs.iter()) {
            assert_eq!(lift(*x, KAHE_MODULUS), lift(*y, KAHE_MODULUS));
        }
    }
}
