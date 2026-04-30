use super::lift;
use super::normalize;
use crate::impl_signed_poly;
use crate::impl_signed_poly_functions;
use crate::Polynomial;
use crate::{
    TerPolyCoeffEncoding, KAHE_MODULUS, KAHE_MODULUS_OVER_TWO, KAHE_SAMPLE_THRESHOLD, N,
};
use rand::Rng;
use sha2::Digest;
use sha2::Sha256;
use std::ops::Mul;

mod kahe_mont_param;
mod kahe_ntt;
mod kahe_ntt_param;
mod simd;

pub use kahe_ntt::{pointwise_dot, KaheNTTPoly};

#[derive(Debug, Clone, Copy)]
// KAHE polynomials in canonical encoding (centered representatives).
pub struct KahePoly {
    pub(crate) coeffs: [i32; N as usize],
}

impl_signed_poly!(KahePoly, KAHE_MODULUS, N);
impl_signed_poly_functions!(
    KahePoly,
    KaheNTTPoly,
    KAHE_MODULUS,
    N,
    KAHE_MODULUS_OVER_TWO,
    KAHE_SAMPLE_THRESHOLD
);

impl Mul for KahePoly {
    type Output = Self;

    // Ring multiplication via NTT.
    fn mul(self, other: Self) -> Self {
        (&(KaheNTTPoly::from(&self) * KaheNTTPoly::from(&other))).into()
    }
}

impl KahePoly {
    /// Construct a KahePoly from a coefficient array (coefficients are taken
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
                c >= -KAHE_MODULUS_OVER_TWO && c <= KAHE_MODULUS_OVER_TWO,
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
            *x = lift(*x, KAHE_MODULUS);
        });
    }
}

#[cfg(test)]
mod test {
    use crate::poly::Polynomial;
    use crate::{KahePoly, KAHE_MODULUS};
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_normalization() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        for _ in 0..10 {
            let mut poly = KahePoly::rand_poly(&mut rng);
            let mut poly2 = poly.clone();

            poly.lift();
            // Add a single random multiple of q ∈ {0, q}. Keeps values within
            // i32 even at KAHE's 30-bit q; lift() must still recover the same
            // canonical representative.
            poly2.coeffs.iter_mut().for_each(|x| {
                if rng.next_u32() & 1 == 1 {
                    *x = (*x as i64 + KAHE_MODULUS as i64) as i32;
                }
            });
            poly2.lift();
            for (e, f) in poly.coeffs.iter().zip(poly2.coeffs.iter()) {
                assert_eq!(e, f)
            }
        }
    }
}
