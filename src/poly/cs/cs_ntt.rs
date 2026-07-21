//! CS multiplication domain. CS q = 139301 is not NTT-friendly (≡ 37 mod
//! 4096), so `CsNTTPoly` holds the forward NTT of the *centered integer*
//! coefficients over the auxiliary prime `p = cs_aux_param::MODULUS`
//! (≡ 1 mod 4096, ≈ 2^58). Pointwise products/sums in this domain compute the
//! exact integer negacyclic convolution as long as the true integer result
//! stays within `(-p/2, p/2)` — guaranteed by the protocol norm bounds
//! (worst case ≈ 2^55.5 for a κ=5 full-range dot). Conversion back to
//! `CsPoly` centers mod p (recovering the exact integer) and reduces mod q.

use super::cs_aux_param as tables;
use super::{normalize, CsPoly};
use crate::poly::ntt64::{self, Ring64};
use crate::{CS_MODULUS, N};

pub(crate) const CS_AUX_RING64: Ring64 = Ring64 {
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

const P: u64 = tables::MODULUS;

#[derive(Debug, Clone, Copy)]
// CS polynomials in the auxiliary-prime NTT encoding, canonical in [0, p).
pub struct CsNTTPoly {
    pub(crate) coeffs: [u64; N as usize],
}

impl Default for CsNTTPoly {
    fn default() -> Self {
        Self {
            coeffs: [0u64; N as usize],
        }
    }
}

impl PartialEq for CsNTTPoly {
    fn eq(&self, rhs: &Self) -> bool {
        self.coeffs == rhs.coeffs
    }
}

impl std::fmt::Display for CsNTTPoly {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{:#16x} {:#16x} {:#16x} {:#16x}",
            self.coeffs[0], self.coeffs[1], self.coeffs[2], self.coeffs[3]
        )
    }
}

impl From<&CsPoly> for CsNTTPoly {
    fn from(poly: &CsPoly) -> Self {
        // Center mod q first (same ring element, minimal magnitude), then
        // embed into Z_p.
        let mut coeffs = [0u64; N];
        for (c, &x) in coeffs.iter_mut().zip(poly.coeffs.iter()) {
            let v = normalize(x, CS_MODULUS) as i64;
            *c = if v < 0 { (P as i64 + v) as u64 } else { v as u64 };
        }
        ntt64::ntt(&CS_AUX_RING64, &mut coeffs);
        Self { coeffs }
    }
}

impl From<CsPoly> for CsNTTPoly {
    fn from(poly: CsPoly) -> Self {
        Self::from(&poly)
    }
}

impl From<&CsNTTPoly> for CsPoly {
    fn from(poly: &CsNTTPoly) -> Self {
        let mut buf = poly.coeffs;
        ntt64::inv_ntt(&CS_AUX_RING64, &mut buf);
        let mut coeffs = [0i32; N];
        for (c, &v) in coeffs.iter_mut().zip(buf.iter()) {
            // Center mod p → the exact integer convolution coefficient.
            let exact = if v > P / 2 {
                v as i64 - P as i64
            } else {
                v as i64
            };
            let q = CS_MODULUS as i64;
            let mut r = exact % q;
            if r > q / 2 {
                r -= q;
            }
            if r < -q / 2 {
                r += q;
            }
            *c = r as i32;
        }
        CsPoly { coeffs }
    }
}

impl From<CsNTTPoly> for CsPoly {
    fn from(poly: CsNTTPoly) -> Self {
        CsPoly::from(&poly)
    }
}

impl std::ops::Add for CsNTTPoly {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut res = self;
        res += other;
        res
    }
}

impl std::ops::AddAssign for CsNTTPoly {
    // Adds the integer preimages; exact while the summed preimage stays
    // within (-p/2, p/2).
    fn add_assign(&mut self, other: Self) {
        self.coeffs
            .iter_mut()
            .zip(other.coeffs)
            .for_each(|(x, y)| *x = ntt64::add_mod(*x, y, P));
    }
}

impl std::ops::Sub for CsNTTPoly {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut res = self;
        res.coeffs
            .iter_mut()
            .zip(other.coeffs)
            .for_each(|(x, y)| *x = ntt64::sub_mod(*x, y, P));
        res
    }
}

impl std::ops::Mul for CsNTTPoly {
    type Output = Self;

    // Coefficient-wise multiplication over the auxiliary-prime NTT domain.
    fn mul(self, other: Self) -> Self {
        let mut res = Self::default();
        for (e, (&f, &g)) in res
            .coeffs
            .iter_mut()
            .zip(self.coeffs.iter().zip(other.coeffs.iter()))
        {
            *e = ntt64::mul_mod(f, g, &CS_AUX_RING64);
        }
        res
    }
}

/// NTT-domain inner product (`Σ a[k] · b[k]`), AVX2-dispatched. Exact while
/// the true integer dot stays within `(-p/2, p/2)`.
pub fn pointwise_dot(a: &[CsNTTPoly], b: &[CsNTTPoly]) -> CsNTTPoly {
    debug_assert_eq!(a.len(), b.len());
    let mut res = CsNTTPoly::default();
    let a_refs: Vec<&[u64; N]> = a.iter().map(|p| &p.coeffs).collect();
    let b_refs: Vec<&[u64; N]> = b.iter().map(|p| &p.coeffs).collect();
    ntt64::pointwise_mac(&CS_AUX_RING64, &mut res.coeffs, &a_refs, &b_refs);
    res
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::poly::Polynomial;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_conversion() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        for _ in 0..10 {
            let poly = CsPoly::rand_poly(&mut rng);
            let poly_ntt: CsNTTPoly = (&poly).into();
            let poly_rec: CsPoly = (&poly_ntt).into();
            assert_eq!(poly, poly_rec);
        }
    }

    #[test]
    fn test_mul_matches_schoolbook() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        for _ in 0..10 {
            let a = CsPoly::rand_poly(&mut rng);
            let b = CsPoly::rand_poly(&mut rng);
            let c: CsPoly = (&(CsNTTPoly::from(&a) * CsNTTPoly::from(&b))).into();
            let c_rec = CsPoly::schoolbook(&a, &b);
            assert_eq!(c, c_rec);
        }
    }

    #[test]
    fn test_dot_matches_schoolbook_sum() {
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        // Full-range operands at κ=5 — the worst-case protocol bound.
        let k = 5;
        let av: Vec<CsPoly> = (0..k).map(|_| CsPoly::rand_poly(&mut rng)).collect();
        let bv: Vec<CsPoly> = (0..k).map(|_| CsPoly::rand_poly(&mut rng)).collect();
        let a_ntt: Vec<CsNTTPoly> = av.iter().map(CsNTTPoly::from).collect();
        let b_ntt: Vec<CsNTTPoly> = bv.iter().map(CsNTTPoly::from).collect();
        let got: CsPoly = (&pointwise_dot(&a_ntt, &b_ntt)).into();
        let mut want = CsPoly::default();
        for (a, b) in av.iter().zip(bv.iter()) {
            want += CsPoly::schoolbook(a, b);
        }
        assert_eq!(got, want);
    }

    #[test]
    fn test_small_norm_mul() {
        // BDLOP-shaped product: uniform a × small-norm r.
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let a = CsPoly::rand_poly(&mut rng);
        let r = CsPoly::rand_mod_p(&mut rng, 116);
        let c: CsPoly = (&(CsNTTPoly::from(&a) * CsNTTPoly::from(&r))).into();
        assert_eq!(c, CsPoly::schoolbook(&a, &r));
    }

    #[test]
    fn test_ntt_domain_linearity() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let a = CsPoly::rand_poly(&mut rng);
        let b = CsPoly::rand_poly(&mut rng);
        let sum_ntt = CsNTTPoly::from(&a) + CsNTTPoly::from(&b);
        let sum: CsPoly = (&sum_ntt).into();
        assert_eq!(sum, a + b);
    }
}
