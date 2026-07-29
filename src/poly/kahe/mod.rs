//! KAHE ring: a 64-bit ring with q = KAHE_MODULUS ≈ 2^48.3. Coefficients are
//! i64 centered representatives; the NTT domain (`KaheNTTPoly`) holds u64
//! canonical values in `[0, q)`.

use crate::{KAHE_MODULUS, KAHE_MODULUS_OVER_TWO, KAHE_SAMPLE_THRESHOLD, N};
use rand::Rng;
use sha2::Digest;
use sha2::Sha256;
use std::ops::Mul;

mod kahe_ntt;
mod kahe_ntt64_param;

pub use kahe_ntt::{pointwise_dot, KaheNTTPoly};
#[cfg(test)]
pub(crate) use kahe_ntt::KAHE_RING64;

#[inline]
pub(crate) fn lift64(a: i64, modulus: i64) -> i64 {
    (a % modulus + modulus) % modulus
}

#[inline]
pub(crate) fn normalize64(a: i64, modulus: i64) -> i64 {
    let mut a = a % modulus;
    if a > modulus / 2 {
        a -= modulus
    }
    if a < -modulus / 2 {
        a += modulus
    }
    a
}

#[derive(Debug, Clone, Copy)]
// KAHE polynomials in canonical encoding (centered representatives).
pub struct KahePoly {
    pub(crate) coeffs: [i64; N as usize],
}

impl PartialEq for KahePoly {
    fn eq(&self, rhs: &Self) -> bool {
        self.coeffs
            .iter()
            .zip(rhs.coeffs.iter())
            .all(|(&x, &y)| lift64(x, KAHE_MODULUS) == lift64(y, KAHE_MODULUS))
    }
}

impl Default for KahePoly {
    fn default() -> Self {
        Self {
            coeffs: [0i64; N as usize],
        }
    }
}

impl std::fmt::Display for KahePoly {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{:#16x} {:#16x} {:#16x} {:#16x}",
            self.coeffs[0], self.coeffs[1], self.coeffs[2], self.coeffs[3]
        )
    }
}

impl std::ops::Add for KahePoly {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut res = self;
        res += other;
        res
    }
}

impl std::ops::AddAssign for KahePoly {
    // Coefficient-wise addition with mod reduction. Centered inputs are
    // ≤ q/2 < 2^49, so i64 sums cannot overflow.
    fn add_assign(&mut self, other: Self) {
        self.coeffs
            .iter_mut()
            .zip(other.coeffs)
            .for_each(|(x, y)| *x = (*x + y) % KAHE_MODULUS)
    }
}

impl std::ops::Sub for KahePoly {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut res = self;
        res -= other;
        res
    }
}

impl std::ops::SubAssign for KahePoly {
    fn sub_assign(&mut self, other: Self) {
        self.coeffs
            .iter_mut()
            .zip(other.coeffs)
            .for_each(|(x, y)| *x = (*x - y) % KAHE_MODULUS)
    }
}

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
    pub fn from_coeffs(coeffs: [i64; N]) -> Self {
        Self { coeffs }
    }

    /// Construct from already-centered signed coefficients (within
    /// `[-q/2, q/2]`). Debug-asserts the bound.
    pub fn from_signed_coeffs(coeffs: &[i64; N]) -> Self {
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
    pub fn coeffs(&self) -> &[i64; N] {
        &self.coeffs
    }

    /// Lift coefficients into `[0, q)` (in-place).
    pub fn lift_inplace(&mut self) {
        self.lift();
    }

    /// Sample a uniformly random polynomial, coefficients centered.
    pub fn rand_poly<R: Rng>(rng: &mut R) -> Self {
        let mut res = Self::default();
        for e in res.coeffs.iter_mut() {
            let mut tmp = rng.next_u64();
            while tmp >= KAHE_SAMPLE_THRESHOLD {
                tmp = rng.next_u64();
            }
            *e = (tmp % KAHE_MODULUS as u64) as i64 - KAHE_MODULUS_OVER_TWO;
        }
        res
    }

    /// Schoolbook negacyclic multiplication — slow, correctness oracle only.
    pub fn schoolbook(a: &Self, b: &Self) -> Self {
        let q = KAHE_MODULUS;
        let mut buf = [0i64; N * 2];
        for i in 0..N {
            for j in 0..N {
                // Per-term reduction keeps partial sums ≤ N·q < 2^61.
                buf[i + j] += ((a.coeffs[i] as i128 * b.coeffs[j] as i128) % q as i128) as i64;
                buf[i + j] %= q;
            }
        }
        let mut c = [0i64; N];
        for i in 0..N {
            c[i] = lift64(buf[i] - buf[i + N], q);
        }
        Self { coeffs: c }
    }

    /// A 32-byte digest of the polynomial (canonical form, 8 LE bytes/coeff).
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for &e in self.coeffs.iter() {
            hasher.update(lift64(e, KAHE_MODULUS).to_le_bytes());
        }
        hasher.finalize().into()
    }

    /// Infinity norm of the polynomial.
    pub fn infinity_norm(&self) -> u64 {
        self.coeffs.iter().map(|&x| x.unsigned_abs()).max().unwrap_or(0)
    }

    /// Lift coefficients into `[0, q)`.
    pub fn lift(&mut self) {
        self.coeffs.iter_mut().for_each(|x| {
            *x = lift64(*x, KAHE_MODULUS);
        });
    }

    /// Center coefficients into `[-q/2, q/2]`.
    pub fn normalize(&mut self) {
        self.coeffs.iter_mut().for_each(|x| {
            *x = normalize64(*x, KAHE_MODULUS);
        });
    }
}

#[cfg(test)]
mod test {
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
            // Add a random multiple of q ∈ {0, q}; lift() must still recover
            // the same canonical representative.
            poly2.coeffs.iter_mut().for_each(|x| {
                if rng.next_u32() & 1 == 1 {
                    *x += KAHE_MODULUS;
                }
            });
            poly2.lift();
            for (e, f) in poly.coeffs.iter().zip(poly2.coeffs.iter()) {
                assert_eq!(e, f)
            }
        }
    }
}
