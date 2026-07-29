use super::kahe_ntt64_param as tables;
use super::{lift64, normalize64, KahePoly};
use crate::poly::ntt64::{self, Ring64};
use crate::{KAHE_MODULUS, KAHE_SAMPLE_THRESHOLD, N};

pub(crate) const KAHE_RING64: Ring64 = Ring64 {
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

#[derive(Debug, Clone, Copy)]
// KAHE polynomials in NTT encoding, coefficients canonical in [0, q).
pub struct KaheNTTPoly {
    pub(crate) coeffs: [u64; N as usize],
}

impl Default for KaheNTTPoly {
    fn default() -> Self {
        Self {
            coeffs: [0u64; N as usize],
        }
    }
}

impl PartialEq for KaheNTTPoly {
    fn eq(&self, rhs: &Self) -> bool {
        self.coeffs == rhs.coeffs
    }
}

impl std::fmt::Display for KaheNTTPoly {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{:#16x} {:#16x} {:#16x} {:#16x}",
            self.coeffs[0], self.coeffs[1], self.coeffs[2], self.coeffs[3]
        )
    }
}

impl From<&KahePoly> for KaheNTTPoly {
    fn from(poly: &KahePoly) -> Self {
        let q = KAHE_RING64.q;
        let mut coeffs = [0u64; N];
        for (c, &x) in coeffs.iter_mut().zip(poly.coeffs.iter()) {
            *c = lift64(x, KAHE_MODULUS) as u64;
        }
        debug_assert!(coeffs.iter().all(|&c| c < q));
        ntt64::ntt(&KAHE_RING64, &mut coeffs);
        Self { coeffs }
    }
}

impl From<KahePoly> for KaheNTTPoly {
    fn from(poly: KahePoly) -> Self {
        Self::from(&poly)
    }
}

impl From<&KaheNTTPoly> for KahePoly {
    fn from(poly: &KaheNTTPoly) -> Self {
        let mut buf = poly.coeffs;
        ntt64::inv_ntt(&KAHE_RING64, &mut buf);
        let mut coeffs = [0i64; N];
        for (c, &x) in coeffs.iter_mut().zip(buf.iter()) {
            *c = normalize64(x as i64, KAHE_MODULUS);
        }
        KahePoly { coeffs }
    }
}

impl From<KaheNTTPoly> for KahePoly {
    fn from(poly: KaheNTTPoly) -> Self {
        KahePoly::from(&poly)
    }
}

impl std::ops::Add for KaheNTTPoly {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut res = self;
        res += other;
        res
    }
}

impl std::ops::AddAssign for KaheNTTPoly {
    fn add_assign(&mut self, other: Self) {
        let q = KAHE_RING64.q;
        self.coeffs
            .iter_mut()
            .zip(other.coeffs)
            .for_each(|(x, y)| *x = ntt64::add_mod(*x, y, q));
    }
}

impl std::ops::Sub for KaheNTTPoly {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let q = KAHE_RING64.q;
        let mut res = self;
        res.coeffs
            .iter_mut()
            .zip(other.coeffs)
            .for_each(|(x, y)| *x = ntt64::sub_mod(*x, y, q));
        res
    }
}

impl std::ops::Mul for KaheNTTPoly {
    type Output = Self;

    // Coefficient-wise multiplication over the NTT domain.
    fn mul(self, other: Self) -> Self {
        let mut res = Self::default();
        for (e, (&f, &g)) in res
            .coeffs
            .iter_mut()
            .zip(self.coeffs.iter().zip(other.coeffs.iter()))
        {
            *e = ntt64::mul_mod(f, g, &KAHE_RING64);
        }
        res
    }
}

impl KaheNTTPoly {
    /// Sample a uniformly random polynomial directly in NTT representation.
    /// Valid because the NTT is a bijection over Z_q^N, so uniform slots give
    /// a uniform ring element.
    pub fn rand_ntt_poly<R: rand::Rng>(rng: &mut R) -> Self {
        let mut res = Self::default();
        for e in res.coeffs.iter_mut() {
            let mut tmp = rng.next_u64();
            while tmp >= KAHE_SAMPLE_THRESHOLD {
                tmp = rng.next_u64();
            }
            *e = tmp % KAHE_MODULUS as u64;
        }
        res
    }
}

/// NTT-domain inner product (`Σ a[k] · b[k]`), AVX2-dispatched.
pub fn pointwise_dot(a: &[KaheNTTPoly], b: &[KaheNTTPoly]) -> KaheNTTPoly {
    debug_assert_eq!(a.len(), b.len());
    let mut res = KaheNTTPoly::default();
    let a_refs: Vec<&[u64; N]> = a.iter().map(|p| &p.coeffs).collect();
    let b_refs: Vec<&[u64; N]> = b.iter().map(|p| &p.coeffs).collect();
    ntt64::pointwise_mac(&KAHE_RING64, &mut res.coeffs, &a_refs, &b_refs);
    res
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_conversion() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        for _ in 0..10 {
            let poly = KahePoly::rand_poly(&mut rng);
            let poly_ntt: KaheNTTPoly = (&poly).into();
            let poly_rec: KahePoly = (&poly_ntt).into();
            assert_eq!(poly, poly_rec);
        }
    }

    #[test]
    fn test_arithmetic() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        for _ in 0..10 {
            let a = KahePoly::rand_poly(&mut rng);
            let a_ntt: KaheNTTPoly = (&a).into();
            let b = KahePoly::rand_poly(&mut rng);
            let b_ntt: KaheNTTPoly = (&b).into();

            let c_ntt = a_ntt * b_ntt;
            let c: KahePoly = (&c_ntt).into();
            let c_rec = KahePoly::schoolbook(&a, &b);
            assert_eq!(c, c_rec);

            let d_ntt = a_ntt + b_ntt;
            let d: KahePoly = (&d_ntt).into();
            let d_rec = a + b;
            assert_eq!(d, d_rec);
        }
    }

    #[test]
    fn test_pointwise_dot() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        let a0 = KahePoly::rand_poly(&mut rng);
        let a1 = KahePoly::rand_poly(&mut rng);
        let b0 = KahePoly::rand_poly(&mut rng);
        let b1 = KahePoly::rand_poly(&mut rng);
        let a_ntt: Vec<KaheNTTPoly> = vec![(&a0).into(), (&a1).into()];
        let b_ntt: Vec<KaheNTTPoly> = vec![(&b0).into(), (&b1).into()];
        let dot = pointwise_dot(&a_ntt, &b_ntt);
        let expected = a_ntt[0] * b_ntt[0] + a_ntt[1] * b_ntt[1];
        assert_eq!(dot, expected);
    }

    #[test]
    fn test_rand_ntt_poly_roundtrip() {
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        let p_ntt = KaheNTTPoly::rand_ntt_poly(&mut rng);
        let p: KahePoly = (&p_ntt).into();
        let back: KaheNTTPoly = (&p).into();
        assert_eq!(p_ntt, back);
    }
}
