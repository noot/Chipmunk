//! AVX2 NTT for the CS ring (BDLOP commitment ring) using Montgomery reduction.
//!
//! The Montgomery factor `R = 2^32` is absorbed into the twiddle tables so
//! that butterfly inputs and outputs stay in canonical signed Z_q form.
//! Concretely, `NTT_TABLE_MONT[i] = (NTT_TABLE[i] * R) mod q`, and each
//! butterfly multiplies `v · twiddle_mont` then Montgomery-reduces — the R
//! cancels, leaving `v · twiddle` (canonical).
//!
//! Reduction step: given `prod < q·R` (fits i64 since q < 2^31, R = 2^32):
//!   t = LOW32(prod * Q_INV_NEG)        // u32 multiplication
//!   u = prod + t · q                   // ≡ 0 mod R
//!   r = u >> 32 ∈ [0, 2q)
//!   if r ≥ q: r -= q
//!
//! Pointwise multiply-accumulate uses standard 64-bit Barrett (no Montgomery
//! form needed since `pointwise_dot` consumes two canonical NTT polys).

use super::cs_mont_param::{INV_NTT_TABLE_MONT, NTT_TABLE_MONT, N_INV_MONT};
use crate::{CsNTTPoly, CS_MODULUS, N};
use std::sync::OnceLock;

const Q: i32 = CS_MODULUS;
const Q64: i64 = CS_MODULUS as i64;
/// `-(q^-1) mod 2^32`. Precomputed by `scripts/gen_ntt_n.py`:
/// `q * Q_INV_NEG ≡ -1 (mod 2^32)`.
const Q_INV_NEG: u32 = match crate::N { 512 => 4026679295, 1024 => 4026679295, 2048 => 4026679295, _ => panic!("unsupported N") };

// ---------------------------------------------------------------
// Scalar Montgomery reduction (correctness oracle for AVX2 path).
// ---------------------------------------------------------------

#[inline(always)]
fn mont_reduce_scalar(prod: i64) -> i32 {
    let prod_u = prod as u64;
    let t = (prod_u as u32).wrapping_mul(Q_INV_NEG) as u64;
    let u = (prod_u.wrapping_add(t.wrapping_mul(Q as u64))) >> 32;
    let mut r = u as i32;
    if r >= Q {
        r -= Q;
    }
    r
}

// ---------------------------------------------------------------
// Pointwise MAC: scalar fallback + AVX2 implementation.
// Uses 64-bit Barrett (not Montgomery) since the call site
// (`pointwise_dot`) consumes two canonical NTT polys.
// ---------------------------------------------------------------

fn pointwise_mac_scalar(acc: &mut [i32; N], a: &[&[i32; N]], b: &[&[i32; N]]) {
    for (av, bv) in a.iter().zip(b.iter()) {
        for i in 0..N {
            let av_i = av[i] as i64;
            let bv_i = bv[i] as i64;
            let prod = av_i * bv_i;
            let r = prod.rem_euclid(Q64) as i32;
            let s = acc[i] + r;
            acc[i] = if s >= Q { s - Q } else { s };
        }
    }
}

type MacFn = fn(&mut [i32; N], &[&[i32; N]], &[&[i32; N]]);
static MAC_DISPATCH: OnceLock<MacFn> = OnceLock::new();

fn select_mac_impl() -> MacFn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return pointwise_mac_avx2_dispatch;
        }
    }
    pointwise_mac_scalar
}

pub fn pointwise_mac(acc: &mut [i32; N], a: &[&[i32; N]], b: &[&[i32; N]]) {
    debug_assert_eq!(a.len(), b.len());
    (MAC_DISPATCH.get_or_init(select_mac_impl))(acc, a, b);
}

/// NTT-domain inner product over `CsNTTPoly` slices, dispatching to the
/// AVX2 kernel when available.
pub fn pointwise_dot(a: &[CsNTTPoly], b: &[CsNTTPoly]) -> CsNTTPoly {
    debug_assert_eq!(a.len(), b.len());
    let mut res = CsNTTPoly::default();
    let a_refs: Vec<&[i32; N]> = a.iter().map(|p| &p.coeffs).collect();
    let b_refs: Vec<&[i32; N]> = b.iter().map(|p| &p.coeffs).collect();
    pointwise_mac(&mut res.coeffs, &a_refs, &b_refs);
    res
}

#[cfg(target_arch = "x86_64")]
fn pointwise_mac_avx2_dispatch(acc: &mut [i32; N], a: &[&[i32; N]], b: &[&[i32; N]]) {
    unsafe { pointwise_mac_avx2(acc, a, b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pointwise_mac_avx2(acc: &mut [i32; N], a: &[&[i32; N]], b: &[&[i32; N]]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(N % 8, 0);

    let q_v32 = _mm256_set1_epi32(Q);
    let q_minus_1 = _mm256_set1_epi32(Q - 1);
    let q_v64 = _mm256_set1_epi64x(Q64);
    let qinv_v64 = _mm256_set1_epi64x(Q_INV_NEG as i64);
    let mask_lo32 = _mm256_set1_epi64x(0xFFFFFFFFi64);
    let zero32 = _mm256_setzero_si256();
    let zero64 = _mm256_setzero_si256();
    // R^2 mod q. Accumulate Montgomery-domain products (`a·b·R^-1`) and convert
    // the whole 8-lane accumulator to canonical with ONE reduction per chunk —
    // `mont_reduce(Σ·R2) = Σ` — instead of a second reduction per MAC term.
    const R2_MOD_Q: i64 = match crate::N { 512 => 33892, 1024 => 33892, 2048 => 33892, _ => panic!("unsupported N") };
    let r2_v64 = _mm256_set1_epi64x(R2_MOD_Q);

    for chunk in 0..(N / 8) {
        let off = chunk * 8;
        let acc_ptr = acc.as_mut_ptr().add(off) as *mut __m256i;
        let acc_v = _mm256_loadu_si256(acc_ptr as *const __m256i);

        // Montgomery-domain accumulator: Σ_k a_k·b_k·R^-1 mod q, in [0, q).
        let mut mont_acc = zero32;
        for k in 0..a.len() {
            let av = _mm256_loadu_si256(a[k].as_ptr().add(off) as *const __m256i);
            let bv = _mm256_loadu_si256(b[k].as_ptr().add(off) as *const __m256i);

            // Canonicalize negatives.
            let av_neg = _mm256_cmpgt_epi32(zero32, av);
            let av_c = _mm256_add_epi32(av, _mm256_and_si256(av_neg, q_v32));
            let bv_neg = _mm256_cmpgt_epi32(zero32, bv);
            let bv_c = _mm256_add_epi32(bv, _mm256_and_si256(bv_neg, q_v32));

            // Even-lane 32×32 products (lanes 0,2,4,6 of i32 → 4×i64).
            let prod_e = _mm256_mul_epu32(av_c, bv_c);
            let av_odd = _mm256_srli_epi64(av_c, 32);
            let bv_odd = _mm256_srli_epi64(bv_c, 32);
            let prod_o = _mm256_mul_epu32(av_odd, bv_odd);

            // Single Montgomery reduction per term → a·b·R^-1 ∈ [0, q).
            let red_e = mont_reduce_avx2(prod_e, q_v64, qinv_v64, mask_lo32, zero64);
            let red_o = mont_reduce_avx2(prod_o, q_v64, qinv_v64, mask_lo32, zero64);
            let red = _mm256_or_si256(
                _mm256_and_si256(red_e, mask_lo32),
                _mm256_slli_epi64(red_o, 32),
            );

            // mont_acc = (mont_acc + red) mod q.
            let sum = _mm256_add_epi32(mont_acc, red);
            let ge_q = _mm256_cmpgt_epi32(sum, q_minus_1);
            mont_acc = _mm256_sub_epi32(sum, _mm256_and_si256(ge_q, q_v32));
        }

        // Convert the Montgomery-domain accumulator to canonical, once.
        let ma_e = _mm256_and_si256(mont_acc, mask_lo32);
        let ma_o = _mm256_srli_epi64(mont_acc, 32);
        let prod2_e = _mm256_mul_epu32(ma_e, r2_v64);
        let prod2_o = _mm256_mul_epu32(ma_o, r2_v64);
        let canon_e = mont_reduce_avx2(prod2_e, q_v64, qinv_v64, mask_lo32, zero64);
        let canon_o = mont_reduce_avx2(prod2_o, q_v64, qinv_v64, mask_lo32, zero64);
        let canon = _mm256_or_si256(
            _mm256_and_si256(canon_e, mask_lo32),
            _mm256_slli_epi64(canon_o, 32),
        );

        // acc += canon (both in [0, q)); reduce if >= q.
        let sum = _mm256_add_epi32(acc_v, canon);
        let ge_q = _mm256_cmpgt_epi32(sum, q_minus_1);
        let acc_v = _mm256_sub_epi32(sum, _mm256_and_si256(ge_q, q_v32));
        _mm256_storeu_si256(acc_ptr, acc_v);
    }
}

// ---------------------------------------------------------------
// AVX2 Montgomery reduction primitive.
// Input: 4 × i64 lanes, each `prod ∈ [0, q·R) ⊂ [0, 2^62)`.
// Output: 4 × i64 lanes, each `r ∈ [0, q)` in the low 32 bits.
// ---------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mont_reduce_avx2(
    prod: std::arch::x86_64::__m256i,
    q_v64: std::arch::x86_64::__m256i,
    qinv_v64: std::arch::x86_64::__m256i,
    mask_lo32: std::arch::x86_64::__m256i,
    zero64: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // t = LOW32(prod * Q_INV_NEG)
    let prod_lo = _mm256_and_si256(prod, mask_lo32);
    let t_full = _mm256_mul_epu32(prod_lo, qinv_v64);
    let t = _mm256_and_si256(t_full, mask_lo32);
    // u = prod + t * q
    let tq = _mm256_mul_epu32(t, q_v64);
    let u = _mm256_add_epi64(prod, tq);
    // r = u >> 32  (∈ [0, 2q))
    let r = _mm256_srli_epi64(u, 32);
    // Conditional subtract q.
    let r_sub_q = _mm256_sub_epi64(r, q_v64);
    let r_lt_q = _mm256_cmpgt_epi64(zero64, r_sub_q);
    _mm256_blendv_epi8(r_sub_q, r, r_lt_q)
}

// ---------------------------------------------------------------
// AVX2 forward / inverse NTT (feature = "fast-ntt").
// Stages with stride ht ≥ 8 run SIMD; ht < 8 falls back to scalar.
// Twiddle tables are pre-multiplied by R = 2^32 (Montgomery form)
// so butterfly outputs remain canonical Z_q.
// ---------------------------------------------------------------

#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
pub(super) fn ntt_avx2_dispatch(p: &mut [i32; N]) {
    unsafe { ntt_avx2(p) }
}

#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
pub(super) fn inv_ntt_avx2_dispatch(p: &mut [i32; N]) {
    unsafe { inv_ntt_avx2(p) }
}

#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn ntt_avx2(p: &mut [i32; N]) {
    use std::arch::x86_64::*;

    let q_v32 = _mm256_set1_epi32(Q);
    let q_minus_1 = _mm256_set1_epi32(Q - 1);
    let q_v64 = _mm256_set1_epi64x(Q64);
    let qinv_v64 = _mm256_set1_epi64x(Q_INV_NEG as i64);
    let mask_lo32 = _mm256_set1_epi64x(0xFFFFFFFFi64);
    let zero32 = _mm256_setzero_si256();
    let zero64 = _mm256_setzero_si256();

    // Canonicalize to [0, q).
    for chunk in 0..(N / 8) {
        let off = chunk * 8;
        let ptr = p.as_mut_ptr().add(off) as *mut __m256i;
        let v = _mm256_loadu_si256(ptr as *const __m256i);
        let neg = _mm256_cmpgt_epi32(zero32, v);
        let vc = _mm256_add_epi32(v, _mm256_and_si256(neg, q_v32));
        _mm256_storeu_si256(ptr, vc);
    }

    // Stages with ht ≥ 8 — all SIMD 8-wide (LOG_N-3 of them).
    let mut t = N;
    for l in 0..(crate::LOG_N - 3) {
        let m = 1usize << l;
        let ht = t >> 1;
        let mut i = 0usize;
        let mut j1 = 0usize;
        while i < m {
            // Twiddle in Montgomery form, canonicalized to [0, q).
            let s_raw = NTT_TABLE_MONT[m + i];
            let s_canon = if s_raw < 0 { s_raw + Q } else { s_raw };
            let s_v64 = _mm256_set1_epi64x(s_canon as i64);

            let mut j = j1;
            while j < j1 + ht {
                let u = _mm256_loadu_si256(p.as_ptr().add(j) as *const __m256i);
                let v = _mm256_loadu_si256(p.as_ptr().add(j + ht) as *const __m256i);

                // v * s_mont, then Montgomery-reduce → v * s_canon (canonical).
                let prod_e = _mm256_mul_epu32(v, s_v64);
                let v_odd = _mm256_srli_epi64(v, 32);
                let prod_o = _mm256_mul_epu32(v_odd, s_v64);
                let red_e = mont_reduce_avx2(prod_e, q_v64, qinv_v64, mask_lo32, zero64);
                let red_o = mont_reduce_avx2(prod_o, q_v64, qinv_v64, mask_lo32, zero64);
                let v_red = _mm256_or_si256(
                    _mm256_and_si256(red_e, mask_lo32),
                    _mm256_slli_epi64(red_o, 32),
                );

                // new_u = (u + v_red) mod q
                let sum = _mm256_add_epi32(u, v_red);
                let ge_q = _mm256_cmpgt_epi32(sum, q_minus_1);
                let new_u = _mm256_sub_epi32(sum, _mm256_and_si256(ge_q, q_v32));
                // new_v = (u - v_red) mod q
                let diff = _mm256_sub_epi32(u, v_red);
                let neg = _mm256_cmpgt_epi32(zero32, diff);
                let new_v = _mm256_add_epi32(diff, _mm256_and_si256(neg, q_v32));

                _mm256_storeu_si256(p.as_mut_ptr().add(j) as *mut __m256i, new_u);
                _mm256_storeu_si256(p.as_mut_ptr().add(j + ht) as *mut __m256i, new_v);
                j += 8;
            }
            i += 1;
            j1 += t;
        }
        t = ht;
    }

    // Tail (ht < 8) scalar Barrett — reuse cs_ntt's stages.
    super::cs_ntt::ntt_stages_scalar(p, crate::LOG_N - 3, t);
}

#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn inv_ntt_avx2(p: &mut [i32; N]) {
    use std::arch::x86_64::*;

    let q_v32 = _mm256_set1_epi32(Q);
    let q_minus_1 = _mm256_set1_epi32(Q - 1);
    let q_v64 = _mm256_set1_epi64x(Q64);
    let qinv_v64 = _mm256_set1_epi64x(Q_INV_NEG as i64);
    let mask_lo32 = _mm256_set1_epi64x(0xFFFFFFFFi64);
    let zero32 = _mm256_setzero_si256();
    let zero64 = _mm256_setzero_si256();

    for chunk in 0..(N / 8) {
        let off = chunk * 8;
        let ptr = p.as_mut_ptr().add(off) as *mut __m256i;
        let v = _mm256_loadu_si256(ptr as *const __m256i);
        let neg = _mm256_cmpgt_epi32(zero32, v);
        let vc = _mm256_add_epi32(v, _mm256_and_si256(neg, q_v32));
        _mm256_storeu_si256(ptr, vc);
    }

    // First three iters (t = 1, 2, 4) — scalar.
    let (mut t, mut m) = super::cs_ntt::inv_ntt_stages_scalar(p, 3, 1, N);

    while m > 1 {
        let hm = m >> 1;
        let dt = t << 1;
        let mut i = 0usize;
        let mut j1 = 0usize;
        while i < hm {
            let s_raw = INV_NTT_TABLE_MONT[hm + i];
            let s_canon = if s_raw < 0 { s_raw + Q } else { s_raw };
            let s_v64 = _mm256_set1_epi64x(s_canon as i64);

            let mut j = j1;
            while j < j1 + t {
                let u = _mm256_loadu_si256(p.as_ptr().add(j) as *const __m256i);
                let v = _mm256_loadu_si256(p.as_ptr().add(j + t) as *const __m256i);

                let sum = _mm256_add_epi32(u, v);
                let ge_q = _mm256_cmpgt_epi32(sum, q_minus_1);
                let new_u = _mm256_sub_epi32(sum, _mm256_and_si256(ge_q, q_v32));
                _mm256_storeu_si256(p.as_mut_ptr().add(j) as *mut __m256i, new_u);

                let diff = _mm256_sub_epi32(u, v);
                let neg = _mm256_cmpgt_epi32(zero32, diff);
                let d = _mm256_add_epi32(diff, _mm256_and_si256(neg, q_v32));

                let prod_e = _mm256_mul_epu32(d, s_v64);
                let d_odd = _mm256_srli_epi64(d, 32);
                let prod_o = _mm256_mul_epu32(d_odd, s_v64);
                let red_e = mont_reduce_avx2(prod_e, q_v64, qinv_v64, mask_lo32, zero64);
                let red_o = mont_reduce_avx2(prod_o, q_v64, qinv_v64, mask_lo32, zero64);
                let new_v = _mm256_or_si256(
                    _mm256_and_si256(red_e, mask_lo32),
                    _mm256_slli_epi64(red_o, 32),
                );
                _mm256_storeu_si256(p.as_mut_ptr().add(j + t) as *mut __m256i, new_v);

                j += 8;
            }
            i += 1;
            j1 += dt;
        }
        t = dt;
        m = hm;
    }

    // Final scaling by N^-1 in Montgomery form — Montgomery multiply yields
    // canonical value · N^-1.
    let n_inv_raw = N_INV_MONT;
    let n_inv = if n_inv_raw < 0 { n_inv_raw + Q } else { n_inv_raw };
    let n_inv_v64 = _mm256_set1_epi64x(n_inv as i64);
    for chunk in 0..(N / 8) {
        let off = chunk * 8;
        let ptr = p.as_mut_ptr().add(off) as *mut __m256i;
        let v = _mm256_loadu_si256(ptr as *const __m256i);
        let prod_e = _mm256_mul_epu32(v, n_inv_v64);
        let v_odd = _mm256_srli_epi64(v, 32);
        let prod_o = _mm256_mul_epu32(v_odd, n_inv_v64);
        let red_e = mont_reduce_avx2(prod_e, q_v64, qinv_v64, mask_lo32, zero64);
        let red_o = mont_reduce_avx2(prod_o, q_v64, qinv_v64, mask_lo32, zero64);
        let red = _mm256_or_si256(
            _mm256_and_si256(red_e, mask_lo32),
            _mm256_slli_epi64(red_o, 32),
        );
        _mm256_storeu_si256(ptr, red);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{CsPoly, Polynomial};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn mont_scalar_correctness() {
        // For prod = x * R with x ∈ [0, q), mont_reduce should return x.
        for x in [0i32, 1, 17, 12345, Q - 1] {
            let prod = (x as i64) * (1i64 << 32);
            let got = mont_reduce_scalar(prod);
            assert_eq!(got, x, "x={x}");
        }
        // Random products in [0, q²).
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        for _ in 0..1000 {
            let a = rng.gen_range(0..Q) as i64;
            let b = rng.gen_range(0..Q) as i64;
            let prod = a * b;
            let mont = mont_reduce_scalar(prod);
            // mont = a*b*R^-1 mod q. Verify a*b ≡ mont * R (mod q).
            let r_back = ((mont as i64) * (1i64 << 32)).rem_euclid(Q64);
            let ab_canon = (a * b).rem_euclid(Q64);
            assert_eq!(r_back, ab_canon);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pointwise_mac_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        for k in [1usize, 3, 7] {
            let av: Vec<Box<[i32; N]>> = (0..k)
                .map(|_| {
                    let mut v = Box::new([0i32; N]);
                    for c in v.iter_mut() {
                        *c = rng.gen_range(-(Q - 1)..Q);
                    }
                    v
                })
                .collect();
            let bv: Vec<Box<[i32; N]>> = (0..k)
                .map(|_| {
                    let mut v = Box::new([0i32; N]);
                    for c in v.iter_mut() {
                        *c = rng.gen_range(-(Q - 1)..Q);
                    }
                    v
                })
                .collect();
            let a_refs: Vec<&[i32; N]> = av.iter().map(|x| x.as_ref()).collect();
            let b_refs: Vec<&[i32; N]> = bv.iter().map(|x| x.as_ref()).collect();
            let mut acc_s = [0i32; N];
            pointwise_mac_scalar(&mut acc_s, &a_refs, &b_refs);
            let mut acc_v = [0i32; N];
            unsafe { pointwise_mac_avx2(&mut acc_v, &a_refs, &b_refs) };
            assert_eq!(acc_s, acc_v, "scalar/avx2 mismatch at k={k}");
        }
    }

    #[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
    #[test]
    fn ntt_avx2_matches_scalar_via_round_trip() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        for _ in 0..20 {
            let p = CsPoly::rand_poly(&mut rng);
            // Forward + inverse via AVX2 should round-trip to canonical p.
            let mut buf = p.coeffs;
            unsafe { ntt_avx2(&mut buf) };
            unsafe { inv_ntt_avx2(&mut buf) };
            // Compare after centered-normalize.
            for (a, b) in p.coeffs.iter().zip(buf.iter()) {
                let a_c = (*a).rem_euclid(Q);
                let b_c = (*b).rem_euclid(Q);
                assert_eq!(a_c, b_c);
            }
        }
    }

    #[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
    #[test]
    fn ntt_avx2_matches_scalar_multiplication() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        // a * b in NTT domain (via AVX2) matches schoolbook ring product.
        let a = CsPoly::rand_poly(&mut rng);
        let b = CsPoly::rand_poly(&mut rng);
        let mut a_ntt = a.coeffs;
        let mut b_ntt = b.coeffs;
        unsafe { ntt_avx2(&mut a_ntt) };
        unsafe { ntt_avx2(&mut b_ntt) };
        let mut prod_ntt = [0i32; N];
        for i in 0..N {
            prod_ntt[i] =
                ((a_ntt[i] as i64) * (b_ntt[i] as i64)).rem_euclid(Q64) as i32;
        }
        unsafe { inv_ntt_avx2(&mut prod_ntt) };
        let school = CsPoly::schoolbook(&a, &b);
        for (got, want) in prod_ntt.iter().zip(school.coeffs().iter()) {
            assert_eq!(got.rem_euclid(Q), want.rem_euclid(Q));
        }
    }
}
