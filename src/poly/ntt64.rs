//! Shared 64-bit negacyclic NTT core, used by the KAHE ring (q ≈ 2^49.3) and
//! the CS auxiliary convolution ring (p ≈ 2^58). Works on `[u64; N]` buffers
//! with values canonical in `[0, q)`.
//!
//! Butterflies use Shoup multiplication: for a precomputed twiddle `w` with
//! companion `w_shoup = floor(w·2^64/q)`,
//!   q̂ = high64(a · w_shoup);  r = a·w − q̂·q  ∈ [0, 2q)
//! computed entirely in wrapping 64-bit arithmetic (the true value fits since
//! r < 2q < 2^63). Pointwise products (no precomputed companion) use
//! Montgomery reduction with R = 2^64.
//!
//! Requirements on the modulus: q odd prime, q < 2^62, q ≡ 1 (mod 2N).

use crate::{LOG_N, N};

pub(crate) struct Ring64 {
    pub q: u64,
    /// `-(q^-1) mod 2^64`.
    pub q_inv_neg: u64,
    /// `2^128 mod q` (Montgomery conversion factor).
    pub r2: u64,
    pub ntt_table: &'static [u64],
    pub ntt_table_shoup: &'static [u64],
    pub inv_ntt_table: &'static [u64],
    pub inv_ntt_table_shoup: &'static [u64],
    pub n_inv: u64,
    pub n_inv_shoup: u64,
}

#[inline(always)]
pub(crate) fn add_mod(a: u64, b: u64, q: u64) -> u64 {
    let s = a + b;
    if s >= q {
        s - q
    } else {
        s
    }
}

#[inline(always)]
pub(crate) fn sub_mod(a: u64, b: u64, q: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + q - b
    }
}

#[inline(always)]
fn shoup_mul(a: u64, w: u64, w_shoup: u64, q: u64) -> u64 {
    let qq = ((a as u128 * w_shoup as u128) >> 64) as u64;
    let r = a.wrapping_mul(w).wrapping_sub(qq.wrapping_mul(q));
    if r >= q {
        r - q
    } else {
        r
    }
}

/// Montgomery reduction: `x < q·2^64` → `x·2^-64 mod q`, canonical.
#[inline(always)]
pub(crate) fn mont_reduce(x: u128, ring: &Ring64) -> u64 {
    let m = (x as u64).wrapping_mul(ring.q_inv_neg);
    let t = ((x + m as u128 * ring.q as u128) >> 64) as u64;
    if t >= ring.q {
        t - ring.q
    } else {
        t
    }
}

/// `a·b·2^-64 mod q` (Montgomery-domain product).
#[inline(always)]
pub(crate) fn mont_mul(a: u64, b: u64, ring: &Ring64) -> u64 {
    mont_reduce(a as u128 * b as u128, ring)
}

/// Canonical modular product `a·b mod q` for canonical inputs.
#[inline(always)]
pub(crate) fn mul_mod(a: u64, b: u64, ring: &Ring64) -> u64 {
    mont_mul(mont_mul(a, b, ring), ring.r2, ring)
}

// ---------------------------------------------------------------
// Scalar NTT.
// ---------------------------------------------------------------

/// Forward butterfly stages starting at `start_l` with `t = N >> start_l`.
/// Inputs canonical `[0, q)`.
fn ntt_stages_scalar(ring: &Ring64, p: &mut [u64; N], start_l: usize, mut t: usize) {
    let q = ring.q;
    for l in start_l..LOG_N {
        let m = 1usize << l;
        let ht = t >> 1;
        let mut i = 0usize;
        let mut j1 = 0usize;
        while i < m {
            let w = ring.ntt_table[m + i];
            let ws = ring.ntt_table_shoup[m + i];
            let j2 = j1 + ht;
            let mut j = j1;
            while j < j2 {
                let u = p[j];
                let v = shoup_mul(p[j + ht], w, ws, q);
                p[j] = add_mod(u, v, q);
                p[j + ht] = sub_mod(u, v, q);
                j += 1;
            }
            i += 1;
            j1 += t;
        }
        t = ht;
    }
}

/// Inverse butterfly stages, `iters` iterations from `(t, m)`. Returns the
/// updated `(t, m)` for resumption. Inputs canonical `[0, q)`.
fn inv_ntt_stages_scalar(
    ring: &Ring64,
    p: &mut [u64; N],
    iters: usize,
    mut t: usize,
    mut m: usize,
) -> (usize, usize) {
    let q = ring.q;
    for _ in 0..iters {
        if m <= 1 {
            break;
        }
        let hm = m >> 1;
        let dt = t << 1;
        let mut i = 0usize;
        let mut j1 = 0usize;
        while i < hm {
            let w = ring.inv_ntt_table[hm + i];
            let ws = ring.inv_ntt_table_shoup[hm + i];
            let j2 = j1 + t;
            let mut j = j1;
            while j < j2 {
                let u = p[j];
                let v = p[j + t];
                p[j] = add_mod(u, v, q);
                p[j + t] = shoup_mul(sub_mod(u, v, q), w, ws, q);
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

fn inv_ntt_final_scale_scalar(ring: &Ring64, p: &mut [u64; N]) {
    for e in p.iter_mut() {
        *e = shoup_mul(*e, ring.n_inv, ring.n_inv_shoup, ring.q);
    }
}

fn ntt_scalar(ring: &Ring64, p: &mut [u64; N]) {
    ntt_stages_scalar(ring, p, 0, N);
}

fn inv_ntt_scalar(ring: &Ring64, p: &mut [u64; N]) {
    inv_ntt_stages_scalar(ring, p, LOG_N, 1, N);
    inv_ntt_final_scale_scalar(ring, p);
}

// ---------------------------------------------------------------
// Public entry points with AVX2 runtime dispatch.
// ---------------------------------------------------------------

#[inline]
fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Forward NTT over canonical `[0, q)` values, in place.
pub(crate) fn ntt(ring: &Ring64, p: &mut [u64; N]) {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        unsafe { avx2::ntt_avx2(ring, p) };
        return;
    }
    ntt_scalar(ring, p);
}

/// Inverse NTT over canonical `[0, q)` values, in place (includes the final
/// `N^-1` scaling).
pub(crate) fn inv_ntt(ring: &Ring64, p: &mut [u64; N]) {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        unsafe { avx2::inv_ntt_avx2(ring, p) };
        return;
    }
    inv_ntt_scalar(ring, p);
}

// ---------------------------------------------------------------
// Pointwise multiply-accumulate (NTT-domain inner products).
// Accumulates Montgomery-domain products, converting to canonical once per
// coefficient at the end: mont(Σ mont(a·b) · r2) = Σ a·b.
// ---------------------------------------------------------------

fn pointwise_mac_scalar(ring: &Ring64, acc: &mut [u64; N], a: &[&[u64; N]], b: &[&[u64; N]]) {
    let q = ring.q;
    let mut mont_acc = [0u64; N];
    for (av, bv) in a.iter().zip(b.iter()) {
        for i in 0..N {
            mont_acc[i] = add_mod(mont_acc[i], mont_mul(av[i], bv[i], ring), q);
        }
    }
    for i in 0..N {
        acc[i] = add_mod(acc[i], mont_mul(mont_acc[i], ring.r2, ring), q);
    }
}

/// `acc[i] += Σ_k a[k][i]·b[k][i] (mod q)` for canonical inputs.
pub(crate) fn pointwise_mac(ring: &Ring64, acc: &mut [u64; N], a: &[&[u64; N]], b: &[&[u64; N]]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        unsafe { avx2::pointwise_mac_avx2(ring, acc, a, b) };
        return;
    }
    pointwise_mac_scalar(ring, acc, a, b);
}

// ---------------------------------------------------------------
// AVX2 kernels (4 × u64 lanes). 64×64 products are decomposed into
// `_mm256_mul_epu32` partial products; all values stay < 2^62 so signed
// 64-bit lane compares are safe for conditional subtracts.
// ---------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{Ring64, LOG_N, N};
    use std::arch::x86_64::*;

    /// High 64 bits of the 128-bit product of two u64 vectors.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn umulhi64(x: __m256i, y: __m256i) -> __m256i {
        let mask = _mm256_set1_epi64x(0xFFFF_FFFF);
        let x0 = _mm256_and_si256(x, mask);
        let x1 = _mm256_srli_epi64(x, 32);
        let y0 = _mm256_and_si256(y, mask);
        let y1 = _mm256_srli_epi64(y, 32);
        let p00 = _mm256_mul_epu32(x0, y0);
        let p01 = _mm256_mul_epu32(x0, y1);
        let p10 = _mm256_mul_epu32(x1, y0);
        let p11 = _mm256_mul_epu32(x1, y1);
        // mid1 = p01 + (p00 >> 32): ≤ (2^64 − 2^33) + 2^32 — no overflow.
        let mid1 = _mm256_add_epi64(p01, _mm256_srli_epi64(p00, 32));
        // mid2 = p10 + lo32(mid1): same bound — no overflow.
        let mid2 = _mm256_add_epi64(p10, _mm256_and_si256(mid1, mask));
        let hi = _mm256_add_epi64(p11, _mm256_srli_epi64(mid1, 32));
        _mm256_add_epi64(hi, _mm256_srli_epi64(mid2, 32))
    }

    /// Low 64 bits of the product of two u64 vectors (wrapping).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mullo64(x: __m256i, y: __m256i) -> __m256i {
        let mask = _mm256_set1_epi64x(0xFFFF_FFFF);
        let x0 = _mm256_and_si256(x, mask);
        let x1 = _mm256_srli_epi64(x, 32);
        let y0 = _mm256_and_si256(y, mask);
        let y1 = _mm256_srli_epi64(y, 32);
        let p00 = _mm256_mul_epu32(x0, y0);
        let cross = _mm256_add_epi64(_mm256_mul_epu32(x0, y1), _mm256_mul_epu32(x1, y0));
        _mm256_add_epi64(p00, _mm256_slli_epi64(cross, 32))
    }

    /// `if v >= q { v - q } else { v }` for lanes < 2^63 (signed compare safe).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn csub(v: __m256i, q_v: __m256i) -> __m256i {
        let d = _mm256_sub_epi64(v, q_v);
        let lt = _mm256_cmpgt_epi64(_mm256_setzero_si256(), d);
        _mm256_blendv_epi8(d, v, lt)
    }

    /// Shoup multiply by a broadcast twiddle: `a·w mod q`, canonical output.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shoup_mul_v(a: __m256i, w_v: __m256i, ws_v: __m256i, q_v: __m256i) -> __m256i {
        let qq = umulhi64(a, ws_v);
        let r = _mm256_sub_epi64(mullo64(a, w_v), mullo64(qq, q_v));
        csub(r, q_v)
    }

    /// `(a + b) mod q`, canonical inputs.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_mod_v(a: __m256i, b: __m256i, q_v: __m256i) -> __m256i {
        csub(_mm256_add_epi64(a, b), q_v)
    }

    /// `(a - b) mod q`, canonical inputs.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_mod_v(a: __m256i, b: __m256i, q_v: __m256i) -> __m256i {
        let d = _mm256_sub_epi64(a, b);
        let neg = _mm256_cmpgt_epi64(_mm256_setzero_si256(), d);
        _mm256_add_epi64(d, _mm256_and_si256(neg, q_v))
    }

    /// Montgomery product `a·b·2^-64 mod q`, canonical output in `[0, q)`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mont_mul_v(a: __m256i, b: __m256i, q_v: __m256i, qinv_v: __m256i) -> __m256i {
        let lo = mullo64(a, b);
        let hi = umulhi64(a, b);
        let m = mullo64(lo, qinv_v);
        let mhi = umulhi64(m, q_v);
        // (x + m·q) >> 64 = hi + mhi + (lo != 0): low words cancel exactly.
        let lo_nz = _mm256_cmpeq_epi64(lo, _mm256_setzero_si256());
        let carry = _mm256_add_epi64(_mm256_set1_epi64x(1), lo_nz); // 1 if lo≠0 else 0
        let t = _mm256_add_epi64(_mm256_add_epi64(hi, mhi), carry);
        csub(t, q_v)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn ntt_avx2(ring: &Ring64, p: &mut [u64; N]) {
        let q_v = _mm256_set1_epi64x(ring.q as i64);
        let mut t = N;
        for l in 0..(LOG_N - 2) {
            let m = 1usize << l;
            let ht = t >> 1;
            let mut i = 0usize;
            let mut j1 = 0usize;
            while i < m {
                let w_v = _mm256_set1_epi64x(ring.ntt_table[m + i] as i64);
                let ws_v = _mm256_set1_epi64x(ring.ntt_table_shoup[m + i] as i64);
                let mut j = j1;
                while j < j1 + ht {
                    let u = _mm256_loadu_si256(p.as_ptr().add(j) as *const __m256i);
                    let v = _mm256_loadu_si256(p.as_ptr().add(j + ht) as *const __m256i);
                    let v_red = shoup_mul_v(v, w_v, ws_v, q_v);
                    let new_u = add_mod_v(u, v_red, q_v);
                    let new_v = sub_mod_v(u, v_red, q_v);
                    _mm256_storeu_si256(p.as_mut_ptr().add(j) as *mut __m256i, new_u);
                    _mm256_storeu_si256(p.as_mut_ptr().add(j + ht) as *mut __m256i, new_v);
                    j += 4;
                }
                i += 1;
                j1 += t;
            }
            t = ht;
        }
        // Tail stages (ht < 4) scalar.
        super::ntt_stages_scalar(ring, p, LOG_N - 2, t);
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn inv_ntt_avx2(ring: &Ring64, p: &mut [u64; N]) {
        let q_v = _mm256_set1_epi64x(ring.q as i64);
        // First two iterations (t = 1, 2) scalar.
        let (mut t, mut m) = super::inv_ntt_stages_scalar(ring, p, 2, 1, N);
        while m > 1 {
            let hm = m >> 1;
            let dt = t << 1;
            let mut i = 0usize;
            let mut j1 = 0usize;
            while i < hm {
                let w_v = _mm256_set1_epi64x(ring.inv_ntt_table[hm + i] as i64);
                let ws_v = _mm256_set1_epi64x(ring.inv_ntt_table_shoup[hm + i] as i64);
                let mut j = j1;
                while j < j1 + t {
                    let u = _mm256_loadu_si256(p.as_ptr().add(j) as *const __m256i);
                    let v = _mm256_loadu_si256(p.as_ptr().add(j + t) as *const __m256i);
                    let new_u = add_mod_v(u, v, q_v);
                    let d = sub_mod_v(u, v, q_v);
                    let new_v = shoup_mul_v(d, w_v, ws_v, q_v);
                    _mm256_storeu_si256(p.as_mut_ptr().add(j) as *mut __m256i, new_u);
                    _mm256_storeu_si256(p.as_mut_ptr().add(j + t) as *mut __m256i, new_v);
                    j += 4;
                }
                i += 1;
                j1 += dt;
            }
            t = dt;
            m = hm;
        }
        let n_inv_v = _mm256_set1_epi64x(ring.n_inv as i64);
        let n_inv_s_v = _mm256_set1_epi64x(ring.n_inv_shoup as i64);
        for chunk in 0..(N / 4) {
            let ptr = p.as_mut_ptr().add(chunk * 4) as *mut __m256i;
            let v = _mm256_loadu_si256(ptr as *const __m256i);
            _mm256_storeu_si256(ptr, shoup_mul_v(v, n_inv_v, n_inv_s_v, q_v));
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn pointwise_mac_avx2(
        ring: &Ring64,
        acc: &mut [u64; N],
        a: &[&[u64; N]],
        b: &[&[u64; N]],
    ) {
        let q_v = _mm256_set1_epi64x(ring.q as i64);
        let qinv_v = _mm256_set1_epi64x(ring.q_inv_neg as i64);
        let r2_v = _mm256_set1_epi64x(ring.r2 as i64);
        for chunk in 0..(N / 4) {
            let off = chunk * 4;
            let acc_ptr = acc.as_mut_ptr().add(off) as *mut __m256i;
            // Montgomery-domain accumulator: Σ_k a_k·b_k·2^-64 mod q.
            let mut mont_acc = _mm256_setzero_si256();
            for k in 0..a.len() {
                let av = _mm256_loadu_si256(a[k].as_ptr().add(off) as *const __m256i);
                let bv = _mm256_loadu_si256(b[k].as_ptr().add(off) as *const __m256i);
                let prod = mont_mul_v(av, bv, q_v, qinv_v);
                mont_acc = add_mod_v(mont_acc, prod, q_v);
            }
            // One conversion to canonical per chunk: mont(Σ · r2) = Σ.
            let canon = mont_mul_v(mont_acc, r2_v, q_v, qinv_v);
            let acc_v = _mm256_loadu_si256(acc_ptr as *const __m256i);
            _mm256_storeu_si256(acc_ptr, add_mod_v(acc_v, canon, q_v));
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn rings() -> Vec<&'static Ring64> {
        vec![
            &crate::poly::kahe::KAHE_RING64,
            &crate::poly::cs::CS_AUX_RING64,
        ]
    }

    #[test]
    fn mont_and_shoup_primitives() {
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        for ring in rings() {
            let q = ring.q;
            for _ in 0..2000 {
                let a = rng.gen_range(0..q);
                let b = rng.gen_range(0..q);
                let want = ((a as u128 * b as u128) % q as u128) as u64;
                assert_eq!(mul_mod(a, b, ring), want);
                let ws = ((b as u128) << 64) / q as u128;
                assert_eq!(shoup_mul(a, b, ws as u64, q), want);
            }
        }
    }

    #[test]
    fn ntt_roundtrip_scalar() {
        let mut rng = ChaCha20Rng::from_seed([8u8; 32]);
        for ring in rings() {
            for _ in 0..5 {
                let mut p = [0u64; N];
                for e in p.iter_mut() {
                    *e = rng.gen_range(0..ring.q);
                }
                let orig = p;
                ntt_scalar(ring, &mut p);
                inv_ntt_scalar(ring, &mut p);
                assert_eq!(p, orig);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn ntt_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = ChaCha20Rng::from_seed([9u8; 32]);
        for ring in rings() {
            for _ in 0..5 {
                let mut p = [0u64; N];
                for e in p.iter_mut() {
                    *e = rng.gen_range(0..ring.q);
                }
                let mut p_s = p;
                let mut p_v = p;
                ntt_scalar(ring, &mut p_s);
                unsafe { avx2::ntt_avx2(ring, &mut p_v) };
                assert_eq!(p_s, p_v, "forward NTT avx2/scalar mismatch");
                inv_ntt_scalar(ring, &mut p_s);
                unsafe { avx2::inv_ntt_avx2(ring, &mut p_v) };
                assert_eq!(p_s, p_v, "inverse NTT avx2/scalar mismatch");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pointwise_mac_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = ChaCha20Rng::from_seed([10u8; 32]);
        for ring in rings() {
            for k in [1usize, 3, 5] {
                let mk = |rng: &mut ChaCha20Rng| {
                    let mut v = Box::new([0u64; N]);
                    for c in v.iter_mut() {
                        *c = rng.gen_range(0..ring.q);
                    }
                    v
                };
                let av: Vec<_> = (0..k).map(|_| mk(&mut rng)).collect();
                let bv: Vec<_> = (0..k).map(|_| mk(&mut rng)).collect();
                let a_refs: Vec<&[u64; N]> = av.iter().map(|x| x.as_ref()).collect();
                let b_refs: Vec<&[u64; N]> = bv.iter().map(|x| x.as_ref()).collect();
                let mut acc_s = [0u64; N];
                pointwise_mac_scalar(ring, &mut acc_s, &a_refs, &b_refs);
                let mut acc_v = [0u64; N];
                unsafe { avx2::pointwise_mac_avx2(ring, &mut acc_v, &a_refs, &b_refs) };
                assert_eq!(acc_s, acc_v, "MAC avx2/scalar mismatch at k={k}");
            }
        }
    }
}
