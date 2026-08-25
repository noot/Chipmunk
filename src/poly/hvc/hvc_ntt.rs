use crate::{normalize, HVCPoly, HVC_MODULUS, N};
use negacyclic_rings::ntt32::{self, Ring32};
use negacyclic_rings::params::{find_psi32, generate_ring32};
use std::fmt;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};
use std::sync::OnceLock;

fn ring() -> &'static Ring32<N> {
    static RING: OnceLock<Ring32<N>> = OnceLock::new();
    RING.get_or_init(|| {
        let q = HVC_MODULUS as u32;
        generate_ring32(q, find_psi32::<N>(q))
    })
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HVCNTTPoly {
    pub(crate) coeffs: [u32; N],
}

impl Default for HVCNTTPoly {
    fn default() -> Self {
        Self { coeffs: [0; N] }
    }
}

impl fmt::Display for HVCNTTPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:#16x} {:#16x} {:#16x} {:#16x}",
            self.coeffs[0], self.coeffs[1], self.coeffs[2], self.coeffs[3]
        )
    }
}

impl From<&HVCPoly> for HVCNTTPoly {
    fn from(poly: &HVCPoly) -> Self {
        let mut coeffs = poly.coeffs.map(|x| x.rem_euclid(HVC_MODULUS) as u32);
        ntt32::ntt(ring(), &mut coeffs);
        Self { coeffs }
    }
}

impl From<HVCPoly> for HVCNTTPoly {
    fn from(poly: HVCPoly) -> Self {
        Self::from(&poly)
    }
}

impl From<&HVCNTTPoly> for HVCPoly {
    fn from(poly: &HVCNTTPoly) -> Self {
        let mut coeffs = poly.coeffs;
        ntt32::inv_ntt(ring(), &mut coeffs);
        HVCPoly {
            coeffs: coeffs.map(|x| normalize(x as i32, HVC_MODULUS)),
        }
    }
}

impl From<HVCNTTPoly> for HVCPoly {
    fn from(poly: HVCNTTPoly) -> Self {
        Self::from(&poly)
    }
}

impl Add for HVCNTTPoly {
    type Output = Self;
    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl AddAssign for HVCNTTPoly {
    fn add_assign(&mut self, other: Self) {
        ntt32::add_assign(ring(), &mut self.coeffs, &other.coeffs);
    }
}

impl Sub for HVCNTTPoly {
    type Output = Self;
    fn sub(mut self, other: Self) -> Self {
        self -= other;
        self
    }
}

impl SubAssign for HVCNTTPoly {
    fn sub_assign(&mut self, other: Self) {
        ntt32::sub_assign(ring(), &mut self.coeffs, &other.coeffs);
    }
}

impl Mul for HVCNTTPoly {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            coeffs: ntt32::pointwise_mul(ring(), &self.coeffs, &other.coeffs),
        }
    }
}

impl HVCNTTPoly {
    pub(crate) fn is_invertible(&self) -> bool {
        self.coeffs.iter().all(|&x| x != 0)
    }
}

pub fn pointwise_dot(a: &[HVCNTTPoly], b: &[HVCNTTPoly]) -> HVCNTTPoly {
    assert_eq!(a.len(), b.len());
    fn coeffs(polys: &[HVCNTTPoly]) -> &[[u32; N]] {
        // HVCNTTPoly is transparent over its coefficient array.
        unsafe { std::slice::from_raw_parts(polys.as_ptr().cast(), polys.len()) }
    }
    HVCNTTPoly {
        coeffs: ntt32::pointwise_dot(ring(), coeffs(a), coeffs(b)),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::poly::Polynomial;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn conversion_and_arithmetic() {
        let mut rng = ChaCha20Rng::from_seed([0; 32]);
        for _ in 0..10 {
            let a = HVCPoly::rand_poly(&mut rng);
            let b = HVCPoly::rand_poly(&mut rng);
            let a_ntt = HVCNTTPoly::from(a);
            let b_ntt = HVCNTTPoly::from(b);
            assert_eq!(HVCPoly::from(a_ntt), a);
            assert_eq!(HVCPoly::from(a_ntt + b_ntt), a + b);
            assert_eq!(HVCPoly::from(a_ntt * b_ntt), HVCPoly::schoolbook(&a, &b));
            assert_eq!(
                HVCPoly::from(pointwise_dot(&[a_ntt, b_ntt], &[b_ntt, a_ntt])),
                HVCPoly::schoolbook(&a, &b) + HVCPoly::schoolbook(&b, &a)
            );
        }
    }
}
