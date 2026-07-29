//! SIMD pointwise modmul-accumulate over length-`N` `i32` vectors mod
//! `HVC_MODULUS`, plus an i16 AVX2 NTT (`ntt_avx2_i16`) usable whenever
//! q < 2^15 so butterfly sums stay within i16 — true at N=512 and N=1024,
//! not at N=2048 (q = 40961). See `SIMD_PLAN.md`.
//!
//! Input coefficients may be any `i32`. Output coefficients are canonical in
//! `[0, HVC_MODULUS)`. The accumulator must be zeroed by the caller; the
//! kernel adds the new sum into it but assumes the existing contents are
//! already in `[0, HVC_MODULUS)`.

use crate::{HVCNTTPoly, HVC_MODULUS, N};
use std::sync::OnceLock;

const Q: i64 = HVC_MODULUS as i64;
// Barrett constants: every HVC q fits in 16 bits, so (q-1)^2 < 2^32 < 2^36.
// m = floor(2^36 / q) so that for x in [0, 2^36), floor(x/q) ∈ {t, t+1}
// where t = (x * m) >> 36.
const BARRETT_K: i32 = 36;
const BARRETT_M: i64 = (1i64 << BARRETT_K) / Q;

type MacFn = fn(&mut [i32; N], &[&[i32; N]], &[&[i32; N]]);

static DISPATCH: OnceLock<MacFn> = OnceLock::new();

fn select_impl() -> MacFn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return pointwise_mac_avx2_dispatch;
        }
    }
    pointwise_mac_scalar
}

/// Compute `acc[i] += Σ_k (a[k][i] * b[k][i]) mod q` for each `i`.
///
/// `acc` must be zeroed (or contain values already in `[0, q)`) on entry.
/// Output coefficients land in `[0, q)`.
pub fn pointwise_mac(acc: &mut [i32; N], a: &[&[i32; N]], b: &[&[i32; N]]) {
    debug_assert_eq!(a.len(), b.len());
    let f = *DISPATCH.get_or_init(select_impl);
    f(acc, a, b);
}

/// NTT-domain inner product: `Σ_k a[k] * b[k]` reduced mod `q`.
/// Higher-level wrapper around `pointwise_mac` that operates on `HVCNTTPoly`s
/// directly. Both inputs must already be in NTT domain. Result is in NTT domain.
pub fn pointwise_dot(a: &[HVCNTTPoly], b: &[HVCNTTPoly]) -> HVCNTTPoly {
    debug_assert_eq!(a.len(), b.len());
    let mut res = HVCNTTPoly::default();
    let a_refs: Vec<&[i32; N]> = a.iter().map(|p| &p.coeffs).collect();
    let b_refs: Vec<&[i32; N]> = b.iter().map(|p| &p.coeffs).collect();
    pointwise_mac(&mut res.coeffs, &a_refs, &b_refs);
    res
}

/// Pointwise sum mod `q` over many length-`N` i32 vectors:
/// `acc[i] = Σ_k v[k][i]` reduced into normalized form `[-q/2, q/2)`, matching
/// `HVCPoly::Add` semantics (the inputs are typically small, so the existing
/// `infinity_norm`-based norm checks expect a centered representation).
///
/// Inputs may carry coefficients in `(-q, q)`. `acc` must start zeroed; output
/// is in `[-q/2, q/2)`.
pub fn pointwise_sum(acc: &mut [i32; N], vs: &[&[i32; N]]) {
    let f = *SUM_DISPATCH.get_or_init(select_sum_impl);
    f(acc, vs);
}

type SumFn = fn(&mut [i32; N], &[&[i32; N]]);
static SUM_DISPATCH: OnceLock<SumFn> = OnceLock::new();

fn select_sum_impl() -> SumFn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return pointwise_sum_avx2_dispatch;
        }
    }
    pointwise_sum_scalar
}

fn pointwise_sum_scalar(acc: &mut [i32; N], vs: &[&[i32; N]]) {
    // Sum into `acc` as canonical `[0, q)`.
    for v in vs.iter() {
        for i in 0..N {
            let mut vi = v[i] % HVC_MODULUS;
            if vi < 0 {
                vi += HVC_MODULUS;
            }
            let s = acc[i] + vi;
            acc[i] = if s >= HVC_MODULUS { s - HVC_MODULUS } else { s };
        }
    }
    // Final pass: normalize to `[-q/2, q/2)` so `infinity_norm` reports the
    // centered magnitude (matches scalar `HVCPoly::Add` representation).
    let half = HVC_MODULUS / 2;
    for i in 0..N {
        if acc[i] > half {
            acc[i] -= HVC_MODULUS;
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn pointwise_sum_avx2_dispatch(acc: &mut [i32; N], vs: &[&[i32; N]]) {
    unsafe { pointwise_sum_avx2(acc, vs) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pointwise_sum_avx2(acc: &mut [i32; N], vs: &[&[i32; N]]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(N % 8, 0);
    // Inputs can be in `(-q, q)` after a single `% q` reduction. We assume the
    // caller has not amplified beyond that; chipmunk's `AddAssign` keeps each
    // intermediate in that range.
    let q_v = _mm256_set1_epi32(HVC_MODULUS);
    let q_minus_1 = _mm256_set1_epi32(HVC_MODULUS - 1);
    let zero = _mm256_setzero_si256();

    for chunk in 0..(N / 8) {
        let off = chunk * 8;
        let acc_ptr = acc.as_mut_ptr().add(off) as *mut __m256i;
        let mut acc_v = _mm256_loadu_si256(acc_ptr as *const __m256i);

        for v in vs.iter() {
            let vv = _mm256_loadu_si256(v.as_ptr().add(off) as *const __m256i);
            // Canonicalize: if vv < 0, add q.
            let neg = _mm256_cmpgt_epi32(zero, vv);
            let vc = _mm256_add_epi32(vv, _mm256_and_si256(neg, q_v));
            // acc += vc; both acc and vc in [0, q), sum in [0, 2q).
            let s = _mm256_add_epi32(acc_v, vc);
            let ge_q = _mm256_cmpgt_epi32(s, q_minus_1);
            acc_v = _mm256_sub_epi32(s, _mm256_and_si256(ge_q, q_v));
        }
        // Final pass: normalize [0, q) to [-q/2, q/2). If acc > q/2, subtract q.
        let half = _mm256_set1_epi32(HVC_MODULUS / 2);
        let gt_half = _mm256_cmpgt_epi32(acc_v, half);
        acc_v = _mm256_sub_epi32(acc_v, _mm256_and_si256(gt_half, q_v));
        _mm256_storeu_si256(acc_ptr, acc_v);
    }
}

/// `HVCPoly`-level wrapper around `pointwise_sum`: returns `Σ vs` as a fresh
/// `HVCPoly` with coefficients in `[0, q)`.
pub fn pointwise_sum_polys(vs: &[&crate::HVCPoly]) -> crate::HVCPoly {
    let mut acc = [0i32; N];
    let refs: Vec<&[i32; N]> = vs.iter().map(|p| &p.coeffs).collect();
    pointwise_sum(&mut acc, &refs);
    crate::HVCPoly::from_coeffs(acc)
}

fn pointwise_mac_scalar(acc: &mut [i32; N], a: &[&[i32; N]], b: &[&[i32; N]]) {
    for (av, bv) in a.iter().zip(b.iter()) {
        for i in 0..N {
            let prod = (av[i] as i64) * (bv[i] as i64);
            let r = prod.rem_euclid(Q) as i32;
            let s = acc[i] + r;
            acc[i] = if s >= HVC_MODULUS { s - HVC_MODULUS } else { s };
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn pointwise_mac_avx2_dispatch(acc: &mut [i32; N], a: &[&[i32; N]], b: &[&[i32; N]]) {
    // SAFETY: dispatcher selected this only when avx2 is detected.
    unsafe { pointwise_mac_avx2(acc, a, b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pointwise_mac_avx2(acc: &mut [i32; N], a: &[&[i32; N]], b: &[&[i32; N]]) {
    use std::arch::x86_64::*;

    debug_assert_eq!(N % 8, 0);
    let q_v32 = _mm256_set1_epi32(HVC_MODULUS);
    let q_minus_1 = _mm256_set1_epi32(HVC_MODULUS - 1);
    let q64 = _mm256_set1_epi64x(Q);
    let m64 = _mm256_set1_epi64x(BARRETT_M);
    let mask_lo32 = _mm256_set1_epi64x(0xFFFFFFFFi64);
    let zero32 = _mm256_setzero_si256();

    for chunk in 0..(N / 8) {
        let off = chunk * 8;
        let acc_ptr = acc.as_mut_ptr().add(off) as *mut __m256i;
        let mut acc_v = _mm256_loadu_si256(acc_ptr as *const __m256i);

        for k in 0..a.len() {
            let av = _mm256_loadu_si256(a[k].as_ptr().add(off) as *const __m256i);
            let bv = _mm256_loadu_si256(b[k].as_ptr().add(off) as *const __m256i);

            // Canonicalize negatives into [0, q): if lane < 0, add q.
            let av_neg = _mm256_cmpgt_epi32(zero32, av);
            let av_c = _mm256_add_epi32(av, _mm256_and_si256(av_neg, q_v32));
            let bv_neg = _mm256_cmpgt_epi32(zero32, bv);
            let bv_c = _mm256_add_epi32(bv, _mm256_and_si256(bv_neg, q_v32));

            // Even-lane products via mul_epu32 (lanes 0,2,4,6 -> 4 x i64).
            let prod_e = _mm256_mul_epu32(av_c, bv_c);
            // Odd-lane products via shifted operands (lanes 1,3,5,7 -> 4 x i64).
            let av_odd = _mm256_srli_epi64(av_c, 32);
            let bv_odd = _mm256_srli_epi64(bv_c, 32);
            let prod_o = _mm256_mul_epu32(av_odd, bv_odd);

            let red_e = barrett_reduce_avx2(prod_e, q64, m64, mask_lo32);
            let red_o = barrett_reduce_avx2(prod_o, q64, m64, mask_lo32);

            // Recombine: red_e holds 4 reduced values in low 32 of each i64 lane;
            // red_o similarly. Pack into one i32x8 by shifting red_o into the high
            // halves and OR-ing with masked red_e.
            let red = _mm256_or_si256(
                _mm256_and_si256(red_e, mask_lo32),
                _mm256_slli_epi64(red_o, 32),
            );

            // acc += red; if acc >= q, subtract q.
            let sum = _mm256_add_epi32(acc_v, red);
            let ge_q = _mm256_cmpgt_epi32(sum, q_minus_1);
            acc_v = _mm256_sub_epi32(sum, _mm256_and_si256(ge_q, q_v32));
        }
        _mm256_storeu_si256(acc_ptr, acc_v);
    }
}

// =============================================================
// AVX2 forward / inverse NTT (feature = "fast-ntt").
// Vectorises stages where the butterfly stride is ≥ 8; falls back to the
// scalar Barrett implementation in `hvc_ntt` for the small-stride tail.
// =============================================================

#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
pub(super) fn ntt_avx2_dispatch(p: &mut [i32; N]) {
    // SAFETY: dispatcher selected this only when avx2 is detected.
    unsafe { ntt_avx2(p) }
}

#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
pub(super) fn inv_ntt_avx2_dispatch(p: &mut [i32; N]) {
    unsafe { inv_ntt_avx2(p) }
}

#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn ntt_avx2(p: &mut [i32; N]) {
    use super::hvc_ntt_param::NTT_TABLE;
    use std::arch::x86_64::*;

    let q_v32 = _mm256_set1_epi32(HVC_MODULUS);
    let q_minus_1 = _mm256_set1_epi32(HVC_MODULUS - 1);
    let q64 = _mm256_set1_epi64x(Q);
    let m64 = _mm256_set1_epi64x(BARRETT_M);
    let mask_lo32 = _mm256_set1_epi64x(0xFFFFFFFFi64);
    let zero32 = _mm256_setzero_si256();

    // Canonicalise the whole array to [0, q).
    for chunk in 0..(N / 8) {
        let off = chunk * 8;
        let ptr = p.as_mut_ptr().add(off) as *mut __m256i;
        let v = _mm256_loadu_si256(ptr as *const __m256i);
        let neg = _mm256_cmpgt_epi32(zero32, v);
        let vc = _mm256_add_epi32(v, _mm256_and_si256(neg, q_v32));
        _mm256_storeu_si256(ptr, vc);
    }

    // Stages with ht ≥ 8 — all vectorisable 8-wide (LOG_N-3 of them).
    let mut t = N;
    for l in 0..(crate::LOG_N - 3) {
        let m = 1usize << l;
        let ht = t >> 1;
        let mut i = 0usize;
        let mut j1 = 0usize;
        while i < m {
            let s_raw = NTT_TABLE[m + i];
            let s_canon = if s_raw < 0 { s_raw + HVC_MODULUS } else { s_raw };
            let s_v64 = _mm256_set1_epi64x(s_canon as i64);

            let mut j = j1;
            while j < j1 + ht {
                let u = _mm256_loadu_si256(p.as_ptr().add(j) as *const __m256i);
                let v = _mm256_loadu_si256(p.as_ptr().add(j + ht) as *const __m256i);

                // v * s mod q via even/odd split.
                let prod_e = _mm256_mul_epu32(v, s_v64);
                let v_odd = _mm256_srli_epi64(v, 32);
                let prod_o = _mm256_mul_epu32(v_odd, s_v64);
                let red_e = barrett_reduce_avx2(prod_e, q64, m64, mask_lo32);
                let red_o = barrett_reduce_avx2(prod_o, q64, m64, mask_lo32);
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

    // Tail: ht < 8 (ht ∈ {4, 2, 1}) scalar Barrett. Inputs already canonical.
    super::hvc_ntt::ntt_stages_scalar(p, crate::LOG_N - 3, t);
}

#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn inv_ntt_avx2(p: &mut [i32; N]) {
    use super::hvc_ntt_param::INV_NTT_TABLE;
    use std::arch::x86_64::*;

    let q_v32 = _mm256_set1_epi32(HVC_MODULUS);
    let q_minus_1 = _mm256_set1_epi32(HVC_MODULUS - 1);
    let q64 = _mm256_set1_epi64x(Q);
    let m64 = _mm256_set1_epi64x(BARRETT_M);
    let mask_lo32 = _mm256_set1_epi64x(0xFFFFFFFFi64);
    let zero32 = _mm256_setzero_si256();

    // Canonicalise the whole array to [0, q).
    for chunk in 0..(N / 8) {
        let off = chunk * 8;
        let ptr = p.as_mut_ptr().add(off) as *mut __m256i;
        let v = _mm256_loadu_si256(ptr as *const __m256i);
        let neg = _mm256_cmpgt_epi32(zero32, v);
        let vc = _mm256_add_epi32(v, _mm256_and_si256(neg, q_v32));
        _mm256_storeu_si256(ptr, vc);
    }

    // First three iters (t ∈ {1, 2, 4}) — scalar.
    let (mut t, mut m) = super::hvc_ntt::inv_ntt_stages_scalar(p, 3, 1, N);

    // Remaining iters with t ≥ 8 — vectorised.
    while m > 1 {
        let hm = m >> 1;
        let dt = t << 1;
        let mut i = 0usize;
        let mut j1 = 0usize;
        while i < hm {
            let s_raw = INV_NTT_TABLE[hm + i];
            let s_canon = if s_raw < 0 { s_raw + HVC_MODULUS } else { s_raw };
            let s_v64 = _mm256_set1_epi64x(s_canon as i64);

            let mut j = j1;
            while j < j1 + t {
                let u = _mm256_loadu_si256(p.as_ptr().add(j) as *const __m256i);
                let v = _mm256_loadu_si256(p.as_ptr().add(j + t) as *const __m256i);

                // p[j] = (u + v) mod q
                let sum = _mm256_add_epi32(u, v);
                let ge_q = _mm256_cmpgt_epi32(sum, q_minus_1);
                let new_u = _mm256_sub_epi32(sum, _mm256_and_si256(ge_q, q_v32));
                _mm256_storeu_si256(p.as_mut_ptr().add(j) as *mut __m256i, new_u);

                // d = (u - v) mod q in [0, q)
                let diff = _mm256_sub_epi32(u, v);
                let neg = _mm256_cmpgt_epi32(zero32, diff);
                let d = _mm256_add_epi32(diff, _mm256_and_si256(neg, q_v32));

                // p[j+t] = d * s mod q
                let prod_e = _mm256_mul_epu32(d, s_v64);
                let d_odd = _mm256_srli_epi64(d, 32);
                let prod_o = _mm256_mul_epu32(d_odd, s_v64);
                let red_e = barrett_reduce_avx2(prod_e, q64, m64, mask_lo32);
                let red_o = barrett_reduce_avx2(prod_o, q64, m64, mask_lo32);
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

    // Final scaling by HVC_ONE_OVER_N — vectorised pointwise mul mod q.
    let inv_n_raw = crate::HVC_ONE_OVER_N;
    let inv_n = if inv_n_raw < 0 { inv_n_raw + HVC_MODULUS } else { inv_n_raw };
    let inv_n_v64 = _mm256_set1_epi64x(inv_n as i64);
    for chunk in 0..(N / 8) {
        let off = chunk * 8;
        let ptr = p.as_mut_ptr().add(off) as *mut __m256i;
        let v = _mm256_loadu_si256(ptr as *const __m256i);
        let prod_e = _mm256_mul_epu32(v, inv_n_v64);
        let v_odd = _mm256_srli_epi64(v, 32);
        let prod_o = _mm256_mul_epu32(v_odd, inv_n_v64);
        let red_e = barrett_reduce_avx2(prod_e, q64, m64, mask_lo32);
        let red_o = barrett_reduce_avx2(prod_o, q64, m64, mask_lo32);
        let red = _mm256_or_si256(
            _mm256_and_si256(red_e, mask_lo32),
            _mm256_slli_epi64(red_o, 32),
        );
        _mm256_storeu_si256(ptr, red);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn barrett_reduce_avx2(
    prod: std::arch::x86_64::__m256i,
    q64: std::arch::x86_64::__m256i,
    m64: std::arch::x86_64::__m256i,
    mask_lo32: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // prod < 2^36, m < 2^18 -> prod*m < 2^54. AVX2 has no 64x64->64 mul, so
    // emulate via 32-halves: prod = pH<<32 + pL.  prod*m = pL*m + (pH*m)<<32.
    // pH < 2^4, pL < 2^32, m < 2^18 -> both partials < 2^50, fit in 64 bits.
    let p_lo = _mm256_and_si256(prod, mask_lo32);
    let p_hi = _mm256_srli_epi64(prod, 32);
    let pm_lo = _mm256_mul_epu32(p_lo, m64);
    let pm_hi = _mm256_mul_epu32(p_hi, m64);
    let pm = _mm256_add_epi64(pm_lo, _mm256_slli_epi64(pm_hi, 32));
    let t = _mm256_srli_epi64(pm, BARRETT_K);

    // r = prod - t*q. t < 2^18, q < 2^18, so t*q < 2^36 (fits in mul_epu32).
    let tq = _mm256_mul_epu32(t, q64);
    let r = _mm256_sub_epi64(prod, tq);

    // r in [0, 2q). Conditional subtract q via blendv on sign of (r - q).
    let r_sub_q = _mm256_sub_epi64(r, q64);
    let zero = _mm256_setzero_si256();
    // r < q ⇔ r_sub_q < 0 ⇔ cmpgt(0, r_sub_q) all-ones.
    let r_lt_q = _mm256_cmpgt_epi64(zero, r_sub_q);
    _mm256_blendv_epi8(r_sub_q, r, r_lt_q)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::HVCNTTPoly;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn rand_vec<R: Rng>(rng: &mut R) -> Box<[i32; N]> {
        let mut v = Box::new([0i32; N]);
        for c in v.iter_mut() {
            // Cover the full input range we expect: `(-q, q)`.
            *c = rng.gen_range(-(HVC_MODULUS - 1)..HVC_MODULUS);
        }
        v
    }

    #[test]
    fn test_kernel_range() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        for k in [1usize, 3, 6, 8] {
            let av: Vec<_> = (0..k).map(|_| rand_vec(&mut rng)).collect();
            let bv: Vec<_> = (0..k).map(|_| rand_vec(&mut rng)).collect();
            let a_refs: Vec<&[i32; N]> = av.iter().map(|x| x.as_ref()).collect();
            let b_refs: Vec<&[i32; N]> = bv.iter().map(|x| x.as_ref()).collect();
            let mut acc = [0i32; N];
            pointwise_mac(&mut acc, &a_refs, &b_refs);
            for &c in acc.iter() {
                assert!((0..HVC_MODULUS).contains(&c), "coeff {c} out of [0, q)");
            }
        }
    }

    #[test]
    fn test_kernel_zero_acc() {
        let mut acc = [0i32; N];
        pointwise_mac(&mut acc, &[], &[]);
        assert!(acc.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_kernel_vs_existing_mul() {
        // For k = 1, pointwise_mac matches HVCNTTPoly::Mul under PartialEq
        // (which compares residue classes via `lift`).
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        for _ in 0..20 {
            let a = rand_vec(&mut rng);
            let b = rand_vec(&mut rng);
            let mut acc = [0i32; N];
            pointwise_mac(&mut acc, &[a.as_ref()], &[b.as_ref()]);

            let pa = HVCNTTPoly { coeffs: *a };
            let pb = HVCNTTPoly { coeffs: *b };
            let expected = pa * pb;
            let got = HVCNTTPoly { coeffs: acc };
            assert_eq!(got, expected);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sum_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        for k in [1usize, 2, 6, 16, 100] {
            let vv: Vec<_> = (0..k).map(|_| rand_vec(&mut rng)).collect();
            let v_refs: Vec<&[i32; N]> = vv.iter().map(|x| x.as_ref()).collect();

            let mut acc_scalar = [0i32; N];
            super::pointwise_sum_scalar(&mut acc_scalar, &v_refs);

            let mut acc_avx = [0i32; N];
            unsafe { super::pointwise_sum_avx2(&mut acc_avx, &v_refs) };

            assert_eq!(acc_scalar, acc_avx, "scalar vs avx2 sum differ at k={k}");
            let half = HVC_MODULUS / 2;
            for &c in acc_scalar.iter() {
                assert!((-half..=half).contains(&c), "coeff {c} outside (-q/2, q/2]");
            }
        }
    }

    #[test]
    fn test_sum_zero_inputs() {
        let mut acc = [0i32; N];
        pointwise_sum(&mut acc, &[]);
        assert!(acc.iter().all(|&x| x == 0));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_kernel_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        for k in [1usize, 3, 6, 8] {
            let av: Vec<_> = (0..k).map(|_| rand_vec(&mut rng)).collect();
            let bv: Vec<_> = (0..k).map(|_| rand_vec(&mut rng)).collect();
            let a_refs: Vec<&[i32; N]> = av.iter().map(|x| x.as_ref()).collect();
            let b_refs: Vec<&[i32; N]> = bv.iter().map(|x| x.as_ref()).collect();

            let mut acc_scalar = [0i32; N];
            super::pointwise_mac_scalar(&mut acc_scalar, &a_refs, &b_refs);

            let mut acc_avx = [0i32; N];
            unsafe { super::pointwise_mac_avx2(&mut acc_avx, &a_refs, &b_refs) };

            assert_eq!(acc_scalar, acc_avx, "scalar vs avx2 differ at k={k}");
        }
    }

    #[test]
    fn test_kernel_accumulates() {
        // pointwise_mac(k inputs) == sum of k separate single-product calls.
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let k = 6;
        let av: Vec<_> = (0..k).map(|_| rand_vec(&mut rng)).collect();
        let bv: Vec<_> = (0..k).map(|_| rand_vec(&mut rng)).collect();
        let a_refs: Vec<&[i32; N]> = av.iter().map(|x| x.as_ref()).collect();
        let b_refs: Vec<&[i32; N]> = bv.iter().map(|x| x.as_ref()).collect();

        let mut combined = [0i32; N];
        pointwise_mac(&mut combined, &a_refs, &b_refs);

        let mut piecewise = [0i32; N];
        for i in 0..k {
            pointwise_mac(&mut piecewise, &a_refs[i..=i], &b_refs[i..=i]);
        }

        let p_combined = HVCNTTPoly { coeffs: combined };
        let p_piecewise = HVCNTTPoly { coeffs: piecewise };
        assert_eq!(p_combined, p_piecewise);
    }
}

// ===============================================================
// i16 AVX2 NTT — viable now that q_cs = 25601 < 2^15.
//
// Twiddles live in Montgomery form at R = 2^16 (precomputed in
// `hvc_mont_i16_param.rs`). Butterfly multiply produces an i32 result
// from i16 × i16; Montgomery-reduce returns i16 in (-q, q), then a
// pair of conditional reductions canonicalises to [-q/2, q/2]. The
// canonical form keeps subsequent add/sub within i16.
//
// Stages with `ht ≥ 16` run SIMD (16 lanes per AVX2 register).
// Tail stages (`ht ∈ {8, 4, 2, 1}`) fall back to i16 scalar code
// which is itself a meaningful speedup over the existing i32 scalar
// tail (`ntt_stages_scalar`).
// ===============================================================

#[cfg(feature = "fast-ntt")]
use super::hvc_mont_i16_param::{
    INV_NTT_TABLE_MONT_I16, NTT_TABLE_MONT_I16, N_INV_MONT_I16, Q_INV_POS_U16,
};

#[cfg(feature = "fast-ntt")]
const Q_I16: i16 = HVC_MODULUS as i16;
#[cfg(feature = "fast-ntt")]
const Q_HALF_I16: i16 = (HVC_MODULUS / 2) as i16;
#[cfg(feature = "fast-ntt")]
const NEG_Q_HALF_I16: i16 = -((HVC_MODULUS / 2) as i16);

/// Scalar Montgomery reduction (i16 oracle). `prod ∈ [-q·R/2, q·R/2)` fits i32.
/// Returns `prod · R^-1 mod q` centered in `[-q/2, q/2)`.
#[cfg(feature = "fast-ntt")]
#[inline(always)]
fn mont_reduce_i16_scalar(prod: i32) -> i16 {
    let qinv = Q_INV_POS_U16 as u32;
    let q = HVC_MODULUS;
    // t = LOW16(prod * qinv). Use u32 wrapping to avoid i32 overflow at the
    // multiply step (which can reach ~2^32 for prod_lo, qinv both near 2^16).
    let prod_lo = prod as u32 & 0xFFFF;
    let t_u32 = prod_lo.wrapping_mul(qinv);
    let t = t_u32 as i16 as i32;
    let u = prod.wrapping_sub(t.wrapping_mul(q));
    // u is divisible by 2^16; (u >> 16) ∈ (-q, q)
    let mut r = (u >> 16) as i16;
    // Reduce to [-q/2, q/2]
    if r > Q_HALF_I16 {
        r -= Q_I16;
    } else if r < NEG_Q_HALF_I16 {
        r += Q_I16;
    }
    r
}

/// Reduce centered: x ∈ (-q, q) → (-q/2, q/2]. Branch-free condition.
#[cfg(feature = "fast-ntt")]
#[inline(always)]
fn center_reduce_i16(x: i16) -> i16 {
    if x > Q_HALF_I16 {
        x - Q_I16
    } else if x < NEG_Q_HALF_I16 {
        x + Q_I16
    } else {
        x
    }
}

#[cfg(feature = "fast-ntt")]
fn ntt_stages_scalar_i16(p: &mut [i16; N], start_l: usize, mut t: usize) {
    for l in start_l..crate::LOG_N {
        let m = 1usize << l;
        let ht = t >> 1;
        let mut i = 0usize;
        let mut j1 = 0usize;
        while i < m {
            let twiddle = NTT_TABLE_MONT_I16[m + i];
            let mut j = j1;
            while j < j1 + ht {
                let u = p[j];
                let v = p[j + ht];
                let prod = (v as i32) * (twiddle as i32);
                let v_red = mont_reduce_i16_scalar(prod);
                // u and v_red are both in (-q/2, q/2], sum/diff in (-q, q).
                p[j] = center_reduce_i16(u + v_red);
                p[j + ht] = center_reduce_i16(u - v_red);
                j += 1;
            }
            i += 1;
            j1 += t;
        }
        t = ht;
    }
}

#[cfg(feature = "fast-ntt")]
fn inv_ntt_stages_scalar_i16(
    p: &mut [i16; N],
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
            let twiddle = INV_NTT_TABLE_MONT_I16[hm + i];
            let mut j = j1;
            while j < j1 + t {
                let u = p[j];
                let v = p[j + t];
                let sum = u + v;
                let diff = u - v;
                p[j] = center_reduce_i16(sum);
                let prod = (diff as i32) * (twiddle as i32);
                p[j + t] = mont_reduce_i16_scalar(prod);
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

/// Forward NTT on i16 buffer using Montgomery reduction throughout.
/// Stages with ht ≥ 16 SIMD-vectorise to 16 lanes/register.
#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
pub(super) unsafe fn ntt_avx2_i16(p: &mut [i16; N]) {
    use std::arch::x86_64::*;
    let q_v = _mm256_set1_epi16(Q_I16);
    let q_half = _mm256_set1_epi16(Q_HALF_I16);
    let neg_q_half = _mm256_set1_epi16(NEG_Q_HALF_I16);
    let qinv = _mm256_set1_epi16(Q_INV_POS_U16 as i16);

    // Helper closure could go here, but const codegen is friendlier when
    // we inline the butterfly manually.

    let mut t = N;
    // SIMD stages: those with ht ≥ 16 (16 i16 lanes/register). There are
    // LOG_N-4 of them (ht from N/2 down to 16); the rest run scalar.
    for l in 0..(crate::LOG_N - 4) {
        let m = 1usize << l;
        let ht = t >> 1;
        let mut i = 0usize;
        let mut j1 = 0usize;
        while i < m {
            let twiddle_scalar = NTT_TABLE_MONT_I16[m + i];
            let twiddle = _mm256_set1_epi16(twiddle_scalar);
            let mut j = j1;
            while j < j1 + ht {
                let u_ptr = p.as_ptr().add(j) as *const __m256i;
                let v_ptr = p.as_ptr().add(j + ht) as *const __m256i;
                let u = _mm256_loadu_si256(u_ptr);
                let v = _mm256_loadu_si256(v_ptr);

                // v_red = MontgomeryReduce(v · twiddle_mont) ∈ (-q, q)
                let prod_lo = _mm256_mullo_epi16(v, twiddle);
                let prod_hi = _mm256_mulhi_epi16(v, twiddle);
                let t_val = _mm256_mullo_epi16(prod_lo, qinv);
                let tq_hi = _mm256_mulhi_epi16(t_val, q_v);
                let mut v_red = _mm256_sub_epi16(prod_hi, tq_hi);
                // Center-reduce v_red into (-q/2, q/2].
                let gt = _mm256_cmpgt_epi16(v_red, q_half);
                v_red = _mm256_sub_epi16(v_red, _mm256_and_si256(gt, q_v));
                let lt = _mm256_cmpgt_epi16(neg_q_half, v_red);
                v_red = _mm256_add_epi16(v_red, _mm256_and_si256(lt, q_v));

                // Butterfly: new_u = u + v_red, new_v = u - v_red.
                let mut new_u = _mm256_add_epi16(u, v_red);
                let mut new_v = _mm256_sub_epi16(u, v_red);
                // Re-center both halves.
                let gt_u = _mm256_cmpgt_epi16(new_u, q_half);
                new_u = _mm256_sub_epi16(new_u, _mm256_and_si256(gt_u, q_v));
                let lt_u = _mm256_cmpgt_epi16(neg_q_half, new_u);
                new_u = _mm256_add_epi16(new_u, _mm256_and_si256(lt_u, q_v));
                let gt_v = _mm256_cmpgt_epi16(new_v, q_half);
                new_v = _mm256_sub_epi16(new_v, _mm256_and_si256(gt_v, q_v));
                let lt_v = _mm256_cmpgt_epi16(neg_q_half, new_v);
                new_v = _mm256_add_epi16(new_v, _mm256_and_si256(lt_v, q_v));

                _mm256_storeu_si256(p.as_mut_ptr().add(j) as *mut __m256i, new_u);
                _mm256_storeu_si256(p.as_mut_ptr().add(j + ht) as *mut __m256i, new_v);
                j += 16;
            }
            i += 1;
            j1 += t;
        }
        t = ht;
    }

    // Tail: stages with ht < 16 (ht ∈ {8, 4, 2, 1}) scalar i16.
    ntt_stages_scalar_i16(p, crate::LOG_N - 4, t);
}

#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
pub(super) unsafe fn inv_ntt_avx2_i16(p: &mut [i16; N]) {
    use std::arch::x86_64::*;
    let q_v = _mm256_set1_epi16(Q_I16);
    let q_half = _mm256_set1_epi16(Q_HALF_I16);
    let neg_q_half = _mm256_set1_epi16(NEG_Q_HALF_I16);
    let qinv = _mm256_set1_epi16(Q_INV_POS_U16 as i16);

    // Scalar tail first: stages with ht < 16, i.e. iters covering m → m/16.
    // Start: t=1, m=N. We need to run scalar until t = 16 (m = N/16 = 32).
    // Iterations needed: log2(16) = 4.
    let (mut t, mut m) = inv_ntt_stages_scalar_i16(p, 4, 1, N);

    // SIMD tail: ht ≥ 16.
    while m > 1 {
        let hm = m >> 1;
        let dt = t << 1;
        let mut i = 0usize;
        let mut j1 = 0usize;
        while i < hm {
            let twiddle = _mm256_set1_epi16(INV_NTT_TABLE_MONT_I16[hm + i]);
            let mut j = j1;
            while j < j1 + t {
                let u_ptr = p.as_ptr().add(j) as *const __m256i;
                let v_ptr = p.as_ptr().add(j + t) as *const __m256i;
                let u = _mm256_loadu_si256(u_ptr);
                let v = _mm256_loadu_si256(v_ptr);

                let sum = _mm256_add_epi16(u, v);
                let diff = _mm256_sub_epi16(u, v);

                // p[j] = center_reduce(sum)
                let mut sum_r = sum;
                let gt = _mm256_cmpgt_epi16(sum_r, q_half);
                sum_r = _mm256_sub_epi16(sum_r, _mm256_and_si256(gt, q_v));
                let lt = _mm256_cmpgt_epi16(neg_q_half, sum_r);
                sum_r = _mm256_add_epi16(sum_r, _mm256_and_si256(lt, q_v));
                _mm256_storeu_si256(p.as_mut_ptr().add(j) as *mut __m256i, sum_r);

                // p[j+t] = MontReduce(diff * twiddle)
                let prod_lo = _mm256_mullo_epi16(diff, twiddle);
                let prod_hi = _mm256_mulhi_epi16(diff, twiddle);
                let t_val = _mm256_mullo_epi16(prod_lo, qinv);
                let tq_hi = _mm256_mulhi_epi16(t_val, q_v);
                let mut v_red = _mm256_sub_epi16(prod_hi, tq_hi);
                let gt2 = _mm256_cmpgt_epi16(v_red, q_half);
                v_red = _mm256_sub_epi16(v_red, _mm256_and_si256(gt2, q_v));
                let lt2 = _mm256_cmpgt_epi16(neg_q_half, v_red);
                v_red = _mm256_add_epi16(v_red, _mm256_and_si256(lt2, q_v));
                _mm256_storeu_si256(p.as_mut_ptr().add(j + t) as *mut __m256i, v_red);

                j += 16;
            }
            i += 1;
            j1 += dt;
        }
        t = dt;
        m = hm;
    }

    // Final scaling by N^-1 in Montgomery form (one Mont-mul per coefficient).
    let n_inv_v = _mm256_set1_epi16(N_INV_MONT_I16);
    for chunk in 0..(N / 16) {
        let off = chunk * 16;
        let ptr = p.as_mut_ptr().add(off) as *mut __m256i;
        let v = _mm256_loadu_si256(ptr as *const __m256i);
        let prod_lo = _mm256_mullo_epi16(v, n_inv_v);
        let prod_hi = _mm256_mulhi_epi16(v, n_inv_v);
        let t_val = _mm256_mullo_epi16(prod_lo, qinv);
        let tq_hi = _mm256_mulhi_epi16(t_val, q_v);
        let mut r = _mm256_sub_epi16(prod_hi, tq_hi);
        let gt = _mm256_cmpgt_epi16(r, q_half);
        r = _mm256_sub_epi16(r, _mm256_and_si256(gt, q_v));
        let lt = _mm256_cmpgt_epi16(neg_q_half, r);
        r = _mm256_add_epi16(r, _mm256_and_si256(lt, q_v));
        _mm256_storeu_si256(ptr, r);
    }
}

/// Convert i32 → i16 by reducing each coefficient mod q to centered.
/// Inputs may be in `(-q, q)` (after a `% q` step); output is `[-q/2, q/2)`.
#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
unsafe fn pack_i32_to_i16(src: &[i32; N], dst: &mut [i16; N]) {
    for i in 0..N {
        let mut x = src[i] % HVC_MODULUS;
        if x >= HVC_MODULUS / 2 {
            x -= HVC_MODULUS;
        } else if x < -HVC_MODULUS / 2 {
            x += HVC_MODULUS;
        }
        dst[i] = x as i16;
    }
}

/// Convert i16 → i32 by sign-extension (output already in centered form).
#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
unsafe fn unpack_i16_to_i32(src: &[i16; N], dst: &mut [i32; N]) {
    for i in 0..N {
        dst[i] = src[i] as i32;
    }
}

/// Public dispatcher: forward NTT via i16 AVX2 path.
#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
pub(super) fn ntt_avx2_i16_dispatch(p: &mut [i32; N]) {
    let mut buf = [0i16; N];
    unsafe {
        pack_i32_to_i16(p, &mut buf);
        ntt_avx2_i16(&mut buf);
        unpack_i16_to_i32(&buf, p);
    }
}

#[cfg(all(feature = "fast-ntt", target_arch = "x86_64"))]
pub(super) fn inv_ntt_avx2_i16_dispatch(p: &mut [i32; N]) {
    let mut buf = [0i16; N];
    unsafe {
        pack_i32_to_i16(p, &mut buf);
        inv_ntt_avx2_i16(&mut buf);
        unpack_i16_to_i32(&buf, p);
    }
}

#[cfg(all(test, feature = "fast-ntt", target_arch = "x86_64"))]
mod test_i16 {
    use super::*;
    use crate::poly::lift;
    use crate::{HVCPoly, Polynomial, HVC_MODULUS};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn i16_ntt_round_trip() {
        // The i16 Montgomery kernel is only valid for q < 2^15; for larger q
        // (e.g. q_hvc = 40961 at N=2048) the production path falls back to the
        // scalar i32 Barrett NTT, so there is nothing to exercise here.
        if HVC_MODULUS >= (1 << 15) {
            return;
        }
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        for _ in 0..20 {
            let p = HVCPoly::rand_poly(&mut rng);
            let mut buf = p.coeffs;
            ntt_avx2_i16_dispatch(&mut buf);
            inv_ntt_avx2_i16_dispatch(&mut buf);
            for (a, b) in p.coeffs.iter().zip(buf.iter()) {
                assert_eq!(lift(*a, HVC_MODULUS), lift(*b, HVC_MODULUS));
            }
        }
    }

    #[test]
    fn i16_ntt_matches_schoolbook() {
        if HVC_MODULUS >= (1 << 15) {
            return;
        }
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = ChaCha20Rng::from_seed([8u8; 32]);
        let a = HVCPoly::rand_poly(&mut rng);
        let b = HVCPoly::rand_poly(&mut rng);
        let mut a_ntt = a.coeffs;
        let mut b_ntt = b.coeffs;
        ntt_avx2_i16_dispatch(&mut a_ntt);
        ntt_avx2_i16_dispatch(&mut b_ntt);
        let mut prod = [0i32; N];
        for i in 0..N {
            prod[i] =
                ((a_ntt[i] as i64) * (b_ntt[i] as i64)).rem_euclid(HVC_MODULUS as i64) as i32;
        }
        inv_ntt_avx2_i16_dispatch(&mut prod);
        let school = HVCPoly::schoolbook(&a, &b);
        for (got, want) in prod.iter().zip(school.coeffs().iter()) {
            assert_eq!(got.rem_euclid(HVC_MODULUS), want.rem_euclid(HVC_MODULUS));
        }
    }
}

#[cfg(all(test, feature = "fast-ntt"))]
mod test_i16_scalar {
    use super::*;
    use crate::poly::lift;
    use crate::{HVCPoly, Polynomial, HVC_MODULUS};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    /// Bypass AVX2: convert i32→i16, run pure scalar i16 NTT, convert back.
    fn ntt_i16_scalar_only(p: &mut [i32; N]) {
        let mut buf = [0i16; N];
        for i in 0..N {
            let mut x = p[i] % HVC_MODULUS;
            if x > HVC_MODULUS / 2 { x -= HVC_MODULUS; }
            else if x < -HVC_MODULUS / 2 { x += HVC_MODULUS; }
            buf[i] = x as i16;
        }
        ntt_stages_scalar_i16(&mut buf, 0, N);
        for i in 0..N { p[i] = buf[i] as i32; }
    }

    fn inv_ntt_i16_scalar_only(p: &mut [i32; N]) {
        let mut buf = [0i16; N];
        for i in 0..N {
            let mut x = p[i] % HVC_MODULUS;
            if x > HVC_MODULUS / 2 { x -= HVC_MODULUS; }
            else if x < -HVC_MODULUS / 2 { x += HVC_MODULUS; }
            buf[i] = x as i16;
        }
        inv_ntt_stages_scalar_i16(&mut buf, crate::LOG_N, 1, N);
        // Final scale by N_INV_MONT_I16
        for i in 0..N {
            let prod = (buf[i] as i32) * (N_INV_MONT_I16 as i32);
            buf[i] = mont_reduce_i16_scalar(prod);
        }
        for i in 0..N { p[i] = buf[i] as i32; }
    }

    #[test]
    fn i16_scalar_round_trip() {
        if HVC_MODULUS >= (1 << 15) {
            return;
        }
        let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
        for _ in 0..5 {
            let p = HVCPoly::rand_poly(&mut rng);
            let mut buf = p.coeffs;
            ntt_i16_scalar_only(&mut buf);
            inv_ntt_i16_scalar_only(&mut buf);
            for (i, (a, b)) in p.coeffs.iter().zip(buf.iter()).enumerate() {
                assert_eq!(lift(*a, HVC_MODULUS), lift(*b, HVC_MODULUS), "i={i}, expected={a}, got={b}");
            }
        }
    }
}
