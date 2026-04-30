mod hots;
mod hvc;

pub use hots::HOTSHash;
pub use hvc::HVCHash;

use crate::poly::Polynomial;
use rand::Rng;

/// Linear, additively-homomorphic hash usable as a Merkle inner-node compressor.
/// Generic over the polynomial ring so callers can swap parameters at compile time.
pub trait LinearHash: Clone + Default {
    type Poly: Polynomial;

    fn init<R: Rng>(rng: &mut R) -> Self;
    fn hash(&self, inputs: &[Self::Poly]) -> Self::Poly;
    /// Decompose `left` and `right` into low-norm pieces, then hash. Homomorphic.
    fn decom_then_hash(&self, left: &Self::Poly, right: &Self::Poly) -> Self::Poly;
    /// Hash already-decomposed inputs (each of length WIDTH).
    fn hash_separate_inputs(&self, left: &[Self::Poly], right: &[Self::Poly]) -> Self::Poly;
}

impl LinearHash for HVCHash {
    type Poly = crate::HVCPoly;

    fn init<R: Rng>(rng: &mut R) -> Self { HVCHash::init(rng) }
    fn hash(&self, inputs: &[Self::Poly]) -> Self::Poly { HVCHash::hash(self, inputs) }
    fn decom_then_hash(&self, l: &Self::Poly, r: &Self::Poly) -> Self::Poly {
        HVCHash::decom_then_hash(self, l, r)
    }
    fn hash_separate_inputs(&self, l: &[Self::Poly], r: &[Self::Poly]) -> Self::Poly {
        HVCHash::hash_separate_inputs(self, l, r)
    }
}
