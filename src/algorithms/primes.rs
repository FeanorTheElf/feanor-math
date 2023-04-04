use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::integer::*;

#[allow(non_snake_case)]
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

#[allow(non_snake_case)]
pub fn enumerate_primes<I>(ZZ: I, B: &El<I>) -> Vec<El<I>> 
    where I: IntegerRingStore
{
    let bound = ZZ.cast::<StaticRing<i128>>(&StaticRing::<i128>::RING, B.clone()) as u64;
    erathostenes(bound).into_iter().map(|p| ZZ.coerce::<StaticRing<i128>>(&StaticRing::<i128>::RING, p as i128)).collect()
}

#[test]
fn test_enumerate_primes() {
    assert_eq!(vec![2], erathostenes(3));
    assert_eq!(vec![2, 3], erathostenes(4));
    assert_eq!(vec![2, 3, 5, 7, 11, 13, 17, 19], erathostenes(20));
}