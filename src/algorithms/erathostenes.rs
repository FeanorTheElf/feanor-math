use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::integer::*;

pub fn erathostenes(B: u64) -> Vec<u64> {
    let mut primes = Vec::new();
    primes.push(2);
    let mut list = Vec::new();
    list.resize((B / 2) as usize, true);
    for i in 1..(B / 2) {
        let n = i * 2 + 1;
        if list[i as usize] {
            primes.push(n);
            for k in 1..((B / 2 - i) / n) {
                let j = k * n + i;
                list[j as usize] = false;
            }
        }
    }
    return primes;
}

pub fn enumerate_primes<I>(ZZ: I, B: &El<I>) -> Vec<El<I>> 
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    let bound = int_cast(ZZ.clone_el(B), StaticRing::<i128>::RING, &ZZ) as u64;
    erathostenes(bound).into_iter().map(|p| int_cast(p as i128, &ZZ, StaticRing::<i128>::RING)).collect()
}

#[test]
fn test_enumerate_primes() {
    assert_eq!(vec![2], erathostenes(3));
    assert_eq!(vec![2, 3], erathostenes(4));
    assert_eq!(vec![2, 3, 5, 7, 11, 13, 17, 19], erathostenes(20));
}