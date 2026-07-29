//! Digest ring: an Ajtai hash domain for *aggregated* KAHE ciphertexts.
//!
//! `DgtNTTPoly` holds the forward NTT of exact-integer coefficients over
//! `q = dgt_param::MODULUS ≈ 2^61` (≡ 1 mod 4096). Its purpose is to let a
//! verifier bind `Σ ct_i` — the unreduced integer sum of a round's ciphertexts
//! — with a collision-resistant hash `h = A · Σ ct_i`.
//!
//! Why a separate, larger modulus than `q_kahe`. The hash input is the
//! *unreduced* sum, so a collision has `‖·‖∞ ≤ ρ·q_kahe ≈ 2^56.5` at ρ = 300.
//! Ring-SIS is only hard when the modulus sits comfortably above that bound, so
//! hashing in `R_{q_kahe}` itself would be vacuous (β ≈ q) and the CS auxiliary
//! prime at 2^58 leaves only a 2^1.5 margin. At 2^61 the ratio is 2^4.5.
//!
//! Everything here is linear, so it commutes with the operations a round
//! already performs: embedding, NTT, Reed–Solomon coding and cross-client
//! summation can be applied in any order. In particular a coded share may be
//! carried in this NTT domain and summed there, with a single inverse transform
//! at the end to recover the integer sum for decryption.
//!
//! Exactness requires the true integer sum to stay within `(-q/2, q/2)`;
//! callers enforce their own `ρ·q_kahe/2` bound via
//! [`DgtNTTPoly::to_centered_coeffs`], which returns the centered
//! representatives so the bound can be checked.

mod dgt_param;

use dgt_param as tables;
use super::kahe::normalize64;
use crate::poly::ntt64::{self, Ring64};
use crate::{KahePoly, KAHE_MODULUS, N};
use rand::Rng;

pub(crate) const DGT_RING64: Ring64 = Ring64 {
    q: tables::MODULUS,
    q_inv_neg: tables::Q_INV_NEG64,
    r2: tables::R2_MOD_Q,
    ntt_table: tables::NTT_TABLE,
    ntt_table_shoup: tables::NTT_TABLE_SHOUP,
    inv_ntt_table: tables::INV_NTT_TABLE,
    inv_ntt_table_shoup: tables::INV_NTT_TABLE_SHOUP,
    n_inv: tables::N_INV,
    n_inv_shoup: tables::N_INV_SHOUP,
};

/// Digest-ring modulus, ≈ 2^61.
pub const DGT_MODULUS: u64 = tables::MODULUS;

const Q: u64 = tables::MODULUS;

/// A digest-ring element in the NTT domain, canonical in `[0, q)`.
#[derive(Debug, Clone, Copy)]
pub struct DgtNTTPoly {
    coeffs: [u64; N],
}

impl Default for DgtNTTPoly {
    fn default() -> Self {
        Self {
            coeffs: [0u64; N],
        }
    }
}

impl PartialEq for DgtNTTPoly {
    fn eq(&self, rhs: &Self) -> bool {
        self.coeffs == rhs.coeffs
    }
}

impl Eq for DgtNTTPoly {}

impl DgtNTTPoly {
    /// Uniform in the NTT domain — statistically the same as `NTT(uniform)` and
    /// saves the transform. Use for the public hash matrix `A`.
    pub fn rand_ntt_poly<R: Rng>(rng: &mut R) -> Self {
        let mut coeffs = [0u64; N];
        for c in coeffs.iter_mut() {
            // Rejection against the largest multiple of q below 2^64 keeps the
            // draw exactly uniform.
            let thr = (u64::MAX / Q) * Q;
            loop {
                let v: u64 = rng.gen();
                if v < thr {
                    *c = v % Q;
                    break;
                }
            }
        }
        Self { coeffs }
    }

    /// Embed a KAHE ciphertext coefficient-wise as exact integers, then
    /// transform. Coefficients are centered mod `q_kahe` first, so the embedded
    /// value is the minimal-magnitude representative and sums of up to
    /// `q/q_kahe` of them stay exact.
    pub fn from_kahe(poly: &KahePoly) -> Self {
        let mut coeffs = [0u64; N];
        for (c, &x) in coeffs.iter_mut().zip(poly.coeffs().iter()) {
            let v = normalize64(x, KAHE_MODULUS);
            *c = if v < 0 { (Q as i64 + v) as u64 } else { v as u64 };
        }
        ntt64::ntt(&DGT_RING64, &mut coeffs);
        Self { coeffs }
    }

    /// Inverse transform, returning centered representatives — the exact
    /// integer coefficients whenever the true values lie in `(-q/2, q/2)`.
    /// Callers check their own norm bound on the result.
    pub fn to_centered_coeffs(&self) -> [i64; N] {
        let mut buf = self.coeffs;
        ntt64::inv_ntt(&DGT_RING64, &mut buf);
        let mut out = [0i64; N];
        for (o, &v) in out.iter_mut().zip(buf.iter()) {
            *o = if v > Q / 2 {
                v as i64 - Q as i64
            } else {
                v as i64
            };
        }
        out
    }

    /// NTT-domain coefficients, canonical in `[0, q)` — the wire form.
    pub fn coeffs(&self) -> &[u64; N] {
        &self.coeffs
    }

    /// Inverse of [`coeffs`](Self::coeffs). Values are reduced, so a malformed
    /// wire encoding cannot produce a non-canonical element.
    pub fn from_raw(coeffs: &[u64; N]) -> Self {
        let mut c = *coeffs;
        for x in c.iter_mut() {
            *x %= Q;
        }
        Self { coeffs: c }
    }
}

impl std::ops::Add for DgtNTTPoly {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut res = self;
        res += other;
        res
    }
}

impl std::ops::AddAssign for DgtNTTPoly {
    fn add_assign(&mut self, other: Self) {
        self.coeffs
            .iter_mut()
            .zip(other.coeffs)
            .for_each(|(x, y)| *x = ntt64::add_mod(*x, y, Q));
    }
}

impl std::ops::Sub for DgtNTTPoly {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut res = self;
        res.coeffs
            .iter_mut()
            .zip(other.coeffs)
            .for_each(|(x, y)| *x = ntt64::sub_mod(*x, y, Q));
        res
    }
}

/// `Σ a[k] · b[k]` in the NTT domain, AVX2-dispatched — one row of the Ajtai
/// hash `A · ct`.
pub fn pointwise_dot(a: &[DgtNTTPoly], b: &[DgtNTTPoly]) -> DgtNTTPoly {
    debug_assert_eq!(a.len(), b.len());
    let mut res = DgtNTTPoly::default();
    let a_refs: Vec<&[u64; N]> = a.iter().map(|p| &p.coeffs).collect();
    let b_refs: Vec<&[u64; N]> = b.iter().map(|p| &p.coeffs).collect();
    ntt64::pointwise_mac(&DGT_RING64, &mut res.coeffs, &a_refs, &b_refs);
    res
}

#[cfg(test)]
mod test {
    use super::*;

    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn kahe_from(vals: &[(usize, i64)]) -> KahePoly {
        let mut c = [0i64; N];
        for &(i, v) in vals {
            c[i] = v;
        }
        KahePoly::from_coeffs(c)
    }

    #[test]
    fn kahe_round_trip_is_exact() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        for _ in 0..8 {
            let p = KahePoly::rand_poly(&mut rng);
            let back = DgtNTTPoly::from_kahe(&p).to_centered_coeffs();
            for (i, &c) in p.coeffs().iter().enumerate() {
                assert_eq!(back[i], normalize64(c, KAHE_MODULUS), "coeff {i}");
            }
        }
    }

    /// The property the protocol relies on: embedding is additive, so summing
    /// in this domain yields the exact *integer* sum, not a sum mod q_kahe.
    #[test]
    fn summation_is_exact_over_the_integers() {
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        let rho = 300;
        let ps: Vec<KahePoly> = (0..rho).map(|_| KahePoly::rand_poly(&mut rng)).collect();
        let mut acc = DgtNTTPoly::default();
        for p in &ps {
            acc += DgtNTTPoly::from_kahe(p);
        }
        let got = acc.to_centered_coeffs();
        for (i, &g) in got.iter().enumerate() {
            let want: i64 = ps
                .iter()
                .map(|p| normalize64(p.coeffs()[i], KAHE_MODULUS))
                .sum();
            assert_eq!(g, want, "coeff {i} (|want| = {})", want.abs());
        }
    }

    #[test]
    fn pointwise_dot_is_linear_in_its_second_argument() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let a: Vec<DgtNTTPoly> = (0..4).map(|_| DgtNTTPoly::rand_ntt_poly(&mut rng)).collect();
        let x: Vec<KahePoly> = (0..4).map(|_| KahePoly::rand_poly(&mut rng)).collect();
        let y: Vec<KahePoly> = (0..4).map(|_| KahePoly::rand_poly(&mut rng)).collect();
        let xn: Vec<DgtNTTPoly> = x.iter().map(DgtNTTPoly::from_kahe).collect();
        let yn: Vec<DgtNTTPoly> = y.iter().map(DgtNTTPoly::from_kahe).collect();
        let sum: Vec<DgtNTTPoly> = xn.iter().zip(&yn).map(|(&p, &q)| p + q).collect();
        assert_eq!(
            pointwise_dot(&a, &sum),
            pointwise_dot(&a, &xn) + pointwise_dot(&a, &yn)
        );
    }

    #[test]
    fn raw_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let p = DgtNTTPoly::rand_ntt_poly(&mut rng);
        assert_eq!(DgtNTTPoly::from_raw(p.coeffs()), p);
        assert!(p.coeffs().iter().all(|&c| c < DGT_MODULUS));
    }

    #[test]
    fn distinct_inputs_hash_apart() {
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        let a: Vec<DgtNTTPoly> = (0..3).map(|_| DgtNTTPoly::rand_ntt_poly(&mut rng)).collect();
        let base: Vec<KahePoly> = (0..3).map(|_| KahePoly::rand_poly(&mut rng)).collect();
        let bn: Vec<DgtNTTPoly> = base.iter().map(DgtNTTPoly::from_kahe).collect();
        let mut bumped = base.clone();
        bumped[1] += kahe_from(&[(7, 1)]);
        let cn: Vec<DgtNTTPoly> = bumped.iter().map(DgtNTTPoly::from_kahe).collect();
        assert_ne!(pointwise_dot(&a, &bn), pointwise_dot(&a, &cn));
    }
}
