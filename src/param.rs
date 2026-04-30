// =================SHARED PARAM===============
/// Degree of the polynomial ring — the single compile-time knob for the ring
/// size. Supported values: `512`, `1024`, `2048`. Each value selects an
/// NTT-friendly modulus set below (every modulus satisfies `q ≡ 1 (mod 2N)`,
/// the condition for a negacyclic NTT of degree `N`). Edit this line to switch.
pub const N: usize = 2048;
/// `log2(N)`. Drives the NTT butterfly stage counts.
pub const LOG_N: usize = N.trailing_zeros() as usize;

/// Compile-time selection over the three supported ring degrees. `panic!` on an
/// unsupported `N` surfaces at const-eval time (i.e. as a build error).
const fn sel_i32(v512: i32, v1024: i32, v2048: i32) -> i32 {
    match N {
        512 => v512,
        1024 => v1024,
        2048 => v2048,
        _ => panic!("unsupported N: set N to 512, 1024, or 2048"),
    }
}
const fn sel_u32(v512: u32, v1024: u32, v2048: u32) -> u32 {
    match N {
        512 => v512,
        1024 => v1024,
        2048 => v2048,
        _ => panic!("unsupported N: set N to 512, 1024, or 2048"),
    }
}
/// Select one of three precomputed twiddle tables by `N`. Used by the per-ring
/// NTT parameter modules so a single `const N` drives the whole library.
pub(crate) const fn pick_slice_i32(
    a: &'static [i32],
    b: &'static [i32],
    c: &'static [i32],
) -> &'static [i32] {
    match N {
        512 => a,
        1024 => b,
        2048 => c,
        _ => panic!("unsupported N: set N to 512, 1024, or 2048"),
    }
}
pub(crate) const fn pick_slice_i16(
    a: &'static [i16],
    b: &'static [i16],
    c: &'static [i16],
) -> &'static [i16] {
    match N {
        512 => a,
        1024 => b,
        2048 => c,
        _ => panic!("unsupported N: set N to 512, 1024, or 2048"),
    }
}

/// Security parameter
pub const SEC_PARAM: usize = 112;
/// non-zero entries in randomizer
pub const ALPHA: usize = 16;
/// Height of the tree
pub const HEIGHT: usize = 5;
/// base of the decomposition: the decomposed poly have coefficients between [-zeta, zeta]
pub const ZETA: u32 = 34;
/// the arity: 2 * zeta + 1
pub const TWO_ZETA_PLUS_ONE: u32 = 69;

// =================HOTS PARAM=================
/// q for large ring, HOTS modulus. Per-N: 512/1024 share 3168257; 2048 uses
/// 3174401 (3168257 ≢ 1 mod 4096, so it lacks a 2N=4096-th root of unity).
/// Both satisfy `(2·ZETA+1)^HOTS_WIDTH = 59^4 > q`.
pub const HOTS_MODULUS: i32 = sel_i32(3168257, 3168257, 3174401);
/// 1/N mod q
pub const HOTS_ONE_OVER_N: i32 = sel_i32(3162069, 3165163, 3172851);
/// (q-1)/2
pub const HOTS_MODULUS_OVER_TWO: i32 = sel_i32(1584128, 1584128, 1587200);
/// number of ring elements during decomposition
/// require (2 * zeta + 1)^HOTS_WIDTH > HOTS_MODULUS
pub const HOTS_WIDTH: usize = 4;
/// the largest multiple of q that is smaller than 2^32
pub const HOTS_SAMPLE_THRESHOLD: u32 = sel_u32(4292988235, 4292988235, 4294964553);
/// the number of polynomials in a decomposed poly
pub const GAMMA: usize = 6;
/// hamming weight of the hash of the message
pub const ALPHA_H: usize = 37;
/// the infinity norm bound for s0
/// i.e., a fresh s_0 has coefficients in [-PHI, PHI]
pub const PHI: u32 = 13;
///the largest multiple of (2 * phi + 1) that is smaller than 2^32
pub const PHI_SAMPLE_THRESHOLD: u32 = 4294967274;
/// norm bound of s_1 = phi * alpha_H
/// i.e., a fresh s_1 has coefficients in [-phi * alpha_H, phi * alpha_H]
pub const PHI_ALPHA_H: u32 = 481;
///the largest multiple of (2 * PHI_ALPHA_H + 1) that is smaller than 2^32
pub const PHI_ALPHA_H_SAMPLE_THRESHOLD: u32 = 4294966518;

// =================HVC PARAM==================
/// q for small ring. Sized < 2^15 so that AVX2 i16 NTT works with eager
/// per-butterfly center-reduce: `u + v_red ∈ [-q, q] ⊂ [-32767, 32767]`.
/// Prime, ≡ 1 mod 2N. Per-N: 512=>25601, 1024=>18433, 2048=>12289 (the prior
/// values lose a 2N-th root of unity as N grows; q shrinks accordingly, which
/// lowers the per-coefficient plaintext/client budget — see q/2 below).
pub const HVC_MODULUS: i32 = sel_i32(25601, 18433, 40961);
/// 1/N mod q
pub const HVC_ONE_OVER_N: i32 = sel_i32(25551, 18415, 40941);
/// (q-1)/2
pub const HVC_MODULUS_OVER_TWO: i32 = sel_i32(12800, 9216, 20480);
/// the largest multiple of q that is smaller than 2^32
pub const HVC_SAMPLE_THRESHOLD: u32 = sel_u32(4294951765, 4294962732, 4294965655);
/// number of ring elements during decomposition
/// require (2 * zeta + 1)^HVC_WIDTH > HVC_MODULUS — 59^3 = 205379 > all HVC q ✓
pub const HVC_WIDTH: usize = 3;

// =================KAHE PARAM=================
/// q for the KAHE ring. Decoupled from HVC's q so that Willow-style RLWE-KAHE
/// gets a larger plaintext budget. Selected so q ≡ 1 (mod 2N), q prime, and
/// q < 2^30 (so q² fits comfortably in i64 during NTT multiplication). Per-N:
/// 512=>1073738753, 1024=>1073707009, 2048=>1073692673.
pub const KAHE_MODULUS: i32 = sel_i32(59393, 1073707009, 271163393);
/// (q-1)/2 for KAHE_MODULUS.
pub const KAHE_MODULUS_OVER_TWO: i32 = sel_i32(29696, 536853504, 135581696);
/// 1/N mod KAHE_MODULUS.
pub const KAHE_ONE_OVER_N: i32 = sel_i32(59277, 1072658467, 271030989);
/// Largest multiple of KAHE_MODULUS that is < 2^32. Used for unbiased
/// rejection sampling of u32 → Z_q.
pub const KAHE_SAMPLE_THRESHOLD: u32 = sel_u32(4294945402, 4294828036, 4067450895);

// =================CS PARAM (BDLOP commitment ring)=================
/// q for the CS (BDLOP) commitment ring, decoupled from HVC's tree-hash ring
/// and KAHE's encryption ring. `q ≡ 1 (mod 4096)` so a single prime serves all
/// supported N. Prime, ≈ 2^17.2 (147457 = 9·2^14 + 1). The values below are
/// regenerated/validated by `scripts/gen_ntt_n.py`.
pub const CS_MODULUS: i32 = sel_i32(147457, 147457, 147457);
/// (q-1)/2 for CS_MODULUS.
pub const CS_MODULUS_OVER_TWO: i32 = sel_i32(73728, 73728, 73728);
/// 1/N mod CS_MODULUS. (placeholder — written by gen_ntt_n.py)
pub const CS_ONE_OVER_N: i32 = sel_i32(147169, 147313, 147385);
/// Largest multiple of CS_MODULUS that is < 2^32. (placeholder — written by gen_ntt_n.py)
pub const CS_SAMPLE_THRESHOLD: u32 = sel_u32(4294832582, 4294832582, 4294832582);

// =================ENCODING PARAM==================
/// the norm bound for alphas and a_star
/// which is beta_open / (2 * zeta)
pub const ENCODING_NORM_BOUND: u32 = 425;
