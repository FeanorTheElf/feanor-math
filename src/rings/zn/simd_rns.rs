use crate::integer::BigIntRing;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::algorithms::eea::*;
use crate::divisibility::DivisibilityRingStore;

const SIMD_LENGTH: usize = 8;
type SimdVector = core::simd::Simd<u64, SIMD_LENGTH>;
const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

pub struct SIMDZnBase {
    moduli: SimdVector,
    inv_moduli: SimdVector,
    modulus: Option<El<BigIntRing>>
}

impl SIMDZnBase {

    pub fn new(moduli: [u64; SIMD_LENGTH]) -> Self {
        
        let total_modulus = ZZbig.prod(moduli.iter().map(|x| ZZbig.coerce(&ZZ, *x as i64)));
        let modulus = if moduli.iter().all(|n| ZZbig.is_one(&signed_gcd(
            ZZbig.coerce(&ZZ, *n as i64), 
            ZZbig.checked_div(&total_modulus, &ZZbig.coerce(&ZZ, *n as i64)).unwrap(), 
            ZZbig
        ))) {
            Some(total_modulus)
        } else {
            None
        };

        let simd_moduli = SimdVector::from_array(moduli);
        let inv_moduli = std::array::from_fn(|i| {
            let n = moduli[i];
            // our representatives should be allowed to grow up to (including) `4 * n`;
            // however we need that a product is `<= 2^84` for the reduction procedure
            // to work
            assert!(n <= (1 << 40));
            let inv_n = (1u128 << 84) / n as u128;
            assert!(inv_n <= u64::MAX as u128);
            inv_n as u64
        });
        let simd_inv_moduli = SimdVector::from_array(inv_moduli);
        return Self {
            inv_moduli: simd_inv_moduli,
            moduli: simd_moduli,
            modulus: modulus
        };
    }

    pub fn rns_moduli_coprime(&self) -> bool {
        self.modulus.is_some()
    }

    fn bounded_reduce()
}

