use crate::{normalize, HOTSPoly, HOTS_MODULUS, N};
use negacyclic_rings::ntt32::{self, Ring32};
use negacyclic_rings::params::{find_psi32, generate_ring32};
use std::fmt;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};
use std::sync::OnceLock;

fn ring() -> &'static Ring32<N> {
    static RING: OnceLock<Ring32<N>> = OnceLock::new();
    RING.get_or_init(|| {
        let q = HOTS_MODULUS as u32;
        generate_ring32(q, find_psi32::<N>(q))
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HOTSNTTPoly {
    pub(crate) coeffs: [u32; N],
}

impl Default for HOTSNTTPoly {
    fn default() -> Self {
        Self { coeffs: [0; N] }
    }
}

impl fmt::Display for HOTSNTTPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:#16x} {:#16x} {:#16x} {:#16x}",
            self.coeffs[0], self.coeffs[1], self.coeffs[2], self.coeffs[3]
        )
    }
}

impl From<&HOTSPoly> for HOTSNTTPoly {
    fn from(poly: &HOTSPoly) -> Self {
        let mut coeffs = poly.coeffs.map(|x| x.rem_euclid(HOTS_MODULUS) as u32);
        ntt32::ntt(ring(), &mut coeffs);
        Self { coeffs }
    }
}

impl From<HOTSPoly> for HOTSNTTPoly {
    fn from(poly: HOTSPoly) -> Self {
        Self::from(&poly)
    }
}

impl From<&HOTSNTTPoly> for HOTSPoly {
    fn from(poly: &HOTSNTTPoly) -> Self {
        let mut coeffs = poly.coeffs;
        ntt32::inv_ntt(ring(), &mut coeffs);
        HOTSPoly {
            coeffs: coeffs.map(|x| normalize(x as i32, HOTS_MODULUS)),
        }
    }
}

impl From<HOTSNTTPoly> for HOTSPoly {
    fn from(poly: HOTSNTTPoly) -> Self {
        Self::from(&poly)
    }
}

impl Add for HOTSNTTPoly {
    type Output = Self;
    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl AddAssign for HOTSNTTPoly {
    fn add_assign(&mut self, other: Self) {
        ntt32::add_assign(ring(), &mut self.coeffs, &other.coeffs);
    }
}

impl Sub for HOTSNTTPoly {
    type Output = Self;
    fn sub(mut self, other: Self) -> Self {
        self -= other;
        self
    }
}

impl SubAssign for HOTSNTTPoly {
    fn sub_assign(&mut self, other: Self) {
        ntt32::sub_assign(ring(), &mut self.coeffs, &other.coeffs);
    }
}

impl Mul for HOTSNTTPoly {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            coeffs: ntt32::pointwise_mul(ring(), &self.coeffs, &other.coeffs),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Polynomial;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn conversion_and_arithmetic() {
        let mut rng = ChaCha20Rng::from_seed([0; 32]);
        for _ in 0..10 {
            let a = HOTSPoly::rand_poly(&mut rng);
            let b = HOTSPoly::rand_poly(&mut rng);
            let a_ntt = HOTSNTTPoly::from(a);
            let b_ntt = HOTSNTTPoly::from(b);
            assert_eq!(HOTSPoly::from(a_ntt), a);
            assert_eq!(HOTSPoly::from(a_ntt + b_ntt), a + b);
            assert_eq!(HOTSPoly::from(a_ntt * b_ntt), HOTSPoly::schoolbook(&a, &b));
        }
    }
}
