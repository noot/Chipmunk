use crate::normalize;
use crate::{impl_ntt_poly, impl_poly, HOTS_ONE_OVER_N};
use crate::{HOTSPoly, HOTS_MODULUS, N};

use super::ntt_param::{INV_NTT_TABLE, NTT_TABLE};

#[derive(Debug, Clone, Copy)]
// HVC polynomials in NTT encoding
pub struct HOTSNTTPoly {
    pub(crate) coeffs: [i32; N as usize],
}

impl_poly!(HOTSNTTPoly, HOTS_MODULUS, N);
impl_ntt_poly!(HOTSPoly, HOTSNTTPoly, HOTS_MODULUS, N, HOTS_ONE_OVER_N);

// =============================================================
// Fast NTT path (feature = "fast-ntt"): scalar Barrett.
// HOTS isn't on flashnet's hot path, so no AVX2 specialisation.
// =============================================================

#[cfg(feature = "fast-ntt")]
const Q: i32 = HOTS_MODULUS;
#[cfg(feature = "fast-ntt")]
const Q64: i64 = HOTS_MODULUS as i64;
// q ≈ 2^21.6, q^2 < 2^44. K=44 covers the multiply outputs (we pre-canonicalise
// the inverse-NTT subtract before multiplying so the multiplicand stays in
// [0, q)). m = floor(2^44/q); q*m fits in i64 (44 + 22 bits = 66 — too wide).
// Use i128 in the Barrett step to stay safe.
#[cfg(feature = "fast-ntt")]
const HOTS_BARRETT_K: u32 = 44;
#[cfg(feature = "fast-ntt")]
const HOTS_BARRETT_M: i64 = (1i64 << HOTS_BARRETT_K) / Q64;

#[cfg(feature = "fast-ntt")]
#[inline(always)]
fn barrett(x: i64) -> i32 {
    let t = (((x as i128) * (HOTS_BARRETT_M as i128)) >> HOTS_BARRETT_K) as i64;
    let r = x - t * Q64;
    let r = if r >= Q64 { r - Q64 } else { r };
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

#[cfg(feature = "fast-ntt")]
fn ntt(p: &mut [i32; N as usize]) {
    for e in p.iter_mut() {
        *e = canon(*e);
    }
    let mut t = N as usize;
    for l in 0..crate::LOG_N {
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

#[cfg(feature = "fast-ntt")]
fn inv_ntt(p: &mut [i32; N as usize]) {
    for e in p.iter_mut() {
        *e = canon(*e);
    }
    let mut t = 1usize;
    let mut m = N as usize;
    while m > 1 {
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
    let inv_n = canon(HOTS_ONE_OVER_N);
    for e in p.iter_mut() {
        *e = barrett((*e as i64) * (inv_n as i64));
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use super::{inv_ntt, ntt};
    use crate::impl_ntt_poly_tests;
    use crate::Polynomial;
    use crate::{poly::lift, HOTSPoly, HOTS_MODULUS};
    use ark_std::{end_timer, start_timer};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    impl_ntt_poly_tests!(HOTSPoly, HOTSNTTPoly, HOTS_MODULUS);
}
