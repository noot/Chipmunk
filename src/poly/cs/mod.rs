use super::lift;
use super::normalize;
use crate::impl_signed_poly;
use crate::impl_signed_poly_functions;
use crate::Polynomial;
use crate::{
    HVCPoly, TerPolyCoeffEncoding, CS_MODULUS, CS_MODULUS_OVER_TWO, CS_SAMPLE_THRESHOLD,
    HVC_WIDTH, N, TWO_ZETA_PLUS_ONE,
};
use rand::Rng;
use sha2::Digest;
use sha2::Sha256;
use std::ops::Mul;

mod cs_aux_param;
mod cs_ntt;

pub use cs_ntt::{pointwise_dot, CsNTTPoly};
#[cfg(test)]
pub(crate) use cs_ntt::CS_AUX_RING64;

#[derive(Debug, Clone, Copy)]
// CS polynomials in canonical encoding (centered representatives).
pub struct CsPoly {
    pub(crate) coeffs: [i32; N],
}

impl_signed_poly!(CsPoly, CS_MODULUS, N);
impl_signed_poly_functions!(
    CsPoly,
    CsNTTPoly,
    CS_MODULUS,
    N,
    CS_MODULUS_OVER_TWO,
    CS_SAMPLE_THRESHOLD
);

impl Mul for CsPoly {
    type Output = Self;

    // Ring multiplication via NTT.
    fn mul(self, other: Self) -> Self {
        (&(CsNTTPoly::from(&self) * CsNTTPoly::from(&other))).into()
    }
}

impl CsPoly {
    /// Construct a CsPoly from a coefficient array (coefficients are taken
    /// as-is; no modular reduction is performed).
    pub fn from_coeffs(coeffs: [i32; N]) -> Self {
        Self { coeffs }
    }

    /// Construct from already-centered signed coefficients (within
    /// `[-q/2, q/2]`). Debug-asserts the bound.
    pub fn from_signed_coeffs(coeffs: &[i32; N]) -> Self {
        #[cfg(debug_assertions)]
        for &c in coeffs.iter() {
            debug_assert!(
                (-CS_MODULUS_OVER_TWO..=CS_MODULUS_OVER_TWO).contains(&c),
                "from_signed_coeffs: coefficient {c} outside [-q/2, q/2]"
            );
        }
        Self { coeffs: *coeffs }
    }

    /// Borrow the coefficient array.
    pub fn coeffs(&self) -> &[i32; N] {
        &self.coeffs
    }

    /// Lift coefficients into `[-q/2, q/2)` (in-place).
    pub fn lift_inplace(&mut self) {
        self.coeffs.iter_mut().for_each(|x| {
            *x = lift(*x, CS_MODULUS);
        });
    }

    /// CS→HVC bridge: base-`(2ζ+1)` signed-digit decomposition of a CS-ring
    /// element (centered mod `CS_MODULUS`) into `HVC_WIDTH` `HVCPoly` digits,
    /// each coefficient in `[-ζ, ζ]`. `(2ζ+1)^HVC_WIDTH = 69^3 = 328509 >
    /// CS_MODULUS = 147457`, so three digits are injective. The digits are tiny
    /// (`|·| ≤ 34 ≪ q_hvc`), so they embed losslessly in the HVC tree-hash ring.
    /// Mirrors `HVCPoly::decompose_r` but over `CS_MODULUS`.
    pub fn decompose_r_to_hvc(&self) -> [HVCPoly; HVC_WIDTH] {
        let mut base: [i32; N] = core::array::from_fn(|i| normalize(self.coeffs[i], CS_MODULUS));
        let mut digits = [[0i32; N]; HVC_WIDTH];
        for digit in digits.iter_mut() {
            for i in 0..N {
                let d = normalize(base[i], TWO_ZETA_PLUS_ONE as i32);
                digit[i] = d;
                base[i] = (base[i] - d) / TWO_ZETA_PLUS_ONE as i32;
            }
        }
        core::array::from_fn(|w| HVCPoly::from_coeffs(digits[w]))
    }

    /// HVC→CS bridge: recompose `HVC_WIDTH` `HVCPoly` digits back to a CS-ring
    /// element, `Σ_j digit_j · (2ζ+1)^j (mod CS_MODULUS)`. Left-inverse of
    /// [`Self::decompose_r_to_hvc`] and **linear in the digits**, so pointwise-summed
    /// opening digits project to the summed leaf value (preserves `sum_openings`).
    pub fn project_r_from_hvc(decomposed: &[HVCPoly]) -> Self {
        debug_assert_eq!(decomposed.len(), HVC_WIDTH);
        let mut res: [i32; N] = *decomposed[HVC_WIDTH - 1].coeffs();
        for poly in decomposed.iter().rev().skip(1) {
            let c = poly.coeffs();
            for i in 0..N {
                res[i] = res[i] * TWO_ZETA_PLUS_ONE as i32 + c[i];
            }
        }
        for r in res.iter_mut() {
            *r = normalize(*r, CS_MODULUS);
        }
        Self::from_coeffs(res)
    }
}

#[cfg(test)]
mod test {
    use crate::poly::Polynomial;
    use crate::{CsPoly, CS_MODULUS};
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_normalization() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        for _ in 0..10 {
            let mut poly = CsPoly::rand_poly(&mut rng);
            let mut poly2 = poly;

            poly.lift();
            // Add a single random multiple of q ∈ {0, q}. Keeps values within
            // i32 even at the largest supported q; lift() must still recover the
            // same canonical representative.
            poly2.coeffs.iter_mut().for_each(|x| {
                if rng.next_u32() & 1 == 1 {
                    *x = (*x as i64 + CS_MODULUS as i64) as i32;
                }
            });
            poly2.lift();
            for (e, f) in poly.coeffs.iter().zip(poly2.coeffs.iter()) {
                assert_eq!(e, f)
            }
        }
    }
}
