use crate::ring::*;
use crate::homomorphism::*;
use crate::integer::*;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::*;

///
/// Uses the square-and-multiply technique to compute the reduction of `power` times `base`
/// w.r.t. the given operation. The operation must be associative to provide correct results.
/// 
/// # Example
/// ```
/// # use feanor_math::algorithms::sqr_mul::generic_abs_square_and_multiply;
/// # use feanor_math::primitive_int::*;
/// let mut mul_count = 0;
/// let mut square_count = 0;
/// // using + instead of *, we can build any number from repeated additions of 1
/// let result = generic_abs_square_and_multiply(
///     1,
///     &120481,
///     StaticRing::<i64>::RING,
///     |x| {
///         square_count += 1;
///         return x + x;
///     },
///     |x, y| {
///         mul_count += 1;
///         return x + y;
///     },
///     0
/// );
/// assert_eq!(120481, result);
/// ```
/// 
pub fn generic_abs_square_and_multiply<T, U, F, H, I>(base: U, power: &El<I>, int_ring: I, mut square: F, mut multiply_base: H, identity: T) -> T
    where I: IntegerRingStore,
        I::Type: IntegerRing,
        F: FnMut(T) -> T, 
        H: FnMut(&U, T) -> T
{
    try_generic_abs_square_and_multiply(base, power, int_ring, |a| Ok(square(a)), |a, b| Ok(multiply_base(a, b)), identity).unwrap_or_else(|x| x)
}

///
/// Uses the square-and-multiply technique to compute the reduction of `power` times `base`
/// w.r.t. the given operation. The operation must be associative to provide correct results.
/// 
/// This function aborts as soon as any operation returns `Err(_)`.
/// 
#[stability::unstable(feature = "enable")]
pub fn try_generic_abs_square_and_multiply<T, U, F, H, I, E>(base: U, power: &El<I>, int_ring: I, mut square: F, mut multiply_base: H, identity: T) -> Result<T, E>
    where I: IntegerRingStore,
        I::Type: IntegerRing,
        F: FnMut(T) -> Result<T, E>, 
        H: FnMut(&U, T) -> Result<T, E>
{
    if int_ring.is_zero(&power) {
        return Ok(identity);
    } else if int_ring.is_one(&power) {
        return multiply_base(&base, identity);
    }

    let mut result = identity;
    for i in (0..=int_ring.abs_highest_set_bit(power).unwrap()).rev() {
        if int_ring.abs_is_bit_set(power, i) {
            result = multiply_base(&base, square(result)?)?;
        } else {
            result = square(result)?;
        }
    }
    return Ok(result);
}

///
/// Raises `base` to the `power`-th power.
/// 
pub fn generic_pow<H, R, S, I>(base: R::Element, power: &El<I>, int_ring: I, hom: &H) -> S::Element
    where R: ?Sized + RingBase, 
        S: ?Sized + RingBase,
        H: Homomorphism<R, S>,
        I: IntegerRingStore,
        I::Type: IntegerRing
{
    let ring = hom.codomain();
    try_generic_abs_square_and_multiply(
        base, 
        power, 
        int_ring, 
        |mut x| {
            ring.square(&mut x);
            Ok(x)
        }, 
        |x, mut y| { 
            hom.mul_assign_map_ref(&mut y, x);
            Ok(y)
        }, 
        ring.one()
    ).unwrap_or_else(|x| x)
}

///
/// Computes the reduction of `power` times `base` w.r.t. the given operation.
/// The operation must be associative to provide correct results.
/// 
/// The used algorithm relies on a decomposition of `power` and a table of small shortest addition 
/// chains to heuristically reduce the number of operations compared to [`generic_abs_square_and_multiply()`].
/// Note that this introduces some overhead, so in cases where the operation is very cheap, prefer
/// [`generic_abs_square_and_multiply()`].
/// 
#[stability::unstable(feature = "enable")]
pub fn generic_pow_shortest_chain_table<T, F, G, H, I, E>(base: T, power: &El<I>, int_ring: I, mut double: G, mut mul: F, mut clone: H, identity: T) -> Result<T, E>
    where I: IntegerRingStore,
        I::Type: IntegerRing,
        F: FnMut(&T, &T) -> Result<T, E>, 
        G: FnMut(&T) -> Result<T, E>, 
        H: FnMut(&T) -> T
{
    assert!(!int_ring.is_neg(power));
    if int_ring.is_zero(&power) {
        return Ok(identity);
    } else if int_ring.is_one(&power) {
        return Ok(base);
    }

    let mut mult_count = 0;

    const LOG2_BOUND: usize = 6;
    const BOUND: usize = 1 << LOG2_BOUND;
    assert!(SHORTEST_ADDITION_CHAINS.len() > BOUND);
    let mut table = Vec::with_capacity(BOUND);
    table.resize_with(BOUND + 1, || None);
    table[0] = Some(identity);
    table[1] = Some(base);

    #[inline(always)]
    fn eval_power_using_table<T, F, G, E>(power: usize, mul: &mut F, double: &mut G, table: &mut Vec<Option<T>>, mult_count: &mut usize) -> Result<(), E>
        where F: FnMut(&T, &T) -> Result<T, E>,
            G: FnMut(&T) -> Result<T, E>, 
    {
        if table[power].is_none() {
            let (i, j) = SHORTEST_ADDITION_CHAINS[power];
            eval_power_using_table(i, mul, double, table, mult_count)?;
            eval_power_using_table(j, mul, double, table, mult_count)?;
            if i == j {
                *mult_count += 1;
                table[power] = Some(double(table[i].as_ref().unwrap())?);
            } else {
                *mult_count += 1;
                table[power] = Some(mul(table[i].as_ref().unwrap(), table[j].as_ref().unwrap())?);
            }
        }
        return Ok(());
    }

    let bitlen = int_ring.abs_highest_set_bit(power).unwrap() + 1;
    if bitlen < LOG2_BOUND {
        let power = int_cast(int_ring.clone_el(&power), StaticRing::<i32>::RING, &int_ring) as usize;
        eval_power_using_table(power, &mut mul, &mut double, &mut table, &mut mult_count)?;
        return Ok(table.into_iter().nth(power).unwrap().unwrap());
    }

    let start_power = (0..LOG2_BOUND).filter(|j| int_ring.abs_is_bit_set(power, *j + bitlen - LOG2_BOUND)).map(|j| 1 << j).sum::<usize>();
    eval_power_using_table(start_power, &mut mul, &mut double, &mut table, &mut mult_count)?;
    let mut current = clone(table[start_power].as_ref().unwrap());

    for i in (0..=(bitlen - LOG2_BOUND)).rev().step_by(LOG2_BOUND).skip(1) {
        for _ in 0..LOG2_BOUND {
            current = double(&current)?;
            mult_count += 1;
        }
        let local_power = (0..LOG2_BOUND).filter(|j| int_ring.abs_is_bit_set(power, *j + i)).map(|j| 1 << j).sum::<usize>();
        if local_power != 0 {
            eval_power_using_table(local_power, &mut mul, &mut double, &mut table, &mut mult_count)?;
            current = mul(&current, table[local_power].as_ref().unwrap())?;
            mult_count += 1;
        }
    }

    if bitlen % LOG2_BOUND != 0 {
        let final_power = (0..(bitlen % LOG2_BOUND)).filter(|j| int_ring.abs_is_bit_set(power, *j)).map(|j| 1 << j).sum::<usize>();
        eval_power_using_table(final_power, &mut mul, &mut double, &mut table, &mut mult_count)?;
        
        for _ in 0..(bitlen % LOG2_BOUND) {
            current = double(&current)?;
            mult_count += 1;
        }
        if final_power != 0 {
            current = mul(&current, table[final_power].as_ref().unwrap())?;
            mult_count += 1;
        }
    }

    debug_assert!(mult_count <= bitlen * 2);

    return Ok(current);
}

// The advantage of numbers < 128 is that the chains are extensions of each other,
// i.e. we can choose each shortest chain such that also all its prefixes are chosen
// shortest chains for corresponding numbers. The becomes impossible for 149.
// data is from http://wwwhomes.uni-bielefeld.de/achim/addition_chain.html
const SHORTEST_ADDITION_CHAINS: [(usize, usize); 65] = [
    (0, 0),
    (1, 0),
    (1, 1),
    (2, 1),
    (2, 2),
    (3, 2),
    (3, 3),
    (5, 2),
    (4, 4),
    (8, 1),
    (5, 5),
    (10, 1),
    (6, 6),
    (9, 4),
    (7, 7),
    (12, 3),
    (8, 8),
    (9, 8),
    (16, 2),
    (18, 1),
    (10, 10),
    (15, 6),
    (11, 11),
    (20, 3),
    (12, 12),
    (17, 8),
    (13, 13),
    (24, 3),
    (14, 14),
    (25, 4),
    (15, 15),
    (28, 3),
    (16, 16),
    (32, 1),
    (17, 17),
    (26, 9),
    (18, 18),
    (36, 1),
    (19, 19),
    (27, 12),
    (20, 20),
    (40, 1),
    (21, 21),
    (34, 9),
    (22, 22),
    (30, 15),
    (23, 23),
    (46, 1),
    (24, 24),
    (33, 16),
    (25, 25),
    (48, 3),
    (26, 26),
    (37, 16),
    (27, 27),
    (54, 1),
    (28, 28),
    (49, 8),
    (29, 29),
    (56, 3),
    (30, 30),
    (52, 9),
    (31, 31),
    (51, 12),
    (32, 32)
];

#[cfg(test)]
use test::Bencher;
#[cfg(test)]
use crate::rings::zn::zn_64;

#[test]
fn test_generic_abs_square_and_multiply() {
    for i in 0..(1 << 16) {
        assert_eq!(Ok(i), try_generic_abs_square_and_multiply::<_, _, _, _, _, !>(1, &i, StaticRing::<i32>::RING, |a| Ok(a * 2), |a, b| Ok(a + b), 0));
    }
}

#[test]
fn test_generic_pow_shortest_chain_table() {
    for i in 0..(1 << 16) {
        assert_eq!(Ok(i), generic_pow_shortest_chain_table::<_, _, _, _, _, !>(1, &i, StaticRing::<i32>::RING, |a| Ok(a * 2), |a, b| Ok(a + b), |a| *a, 0));
    }
}

#[test]
fn test_shortest_addition_chain_table() {
    for i in 0..SHORTEST_ADDITION_CHAINS.len() {
        println!("{}", i);
        assert_eq!(i, SHORTEST_ADDITION_CHAINS[i].0 + SHORTEST_ADDITION_CHAINS[i].1);
    }
}

#[bench]
fn bench_standard_square_and_multiply(bencher: &mut Bencher) {
    let ring = zn_64::Zn::new(536903681);
    let x = ring.int_hom().map(2);
    bencher.iter(|| {
        assert_el_eq!(&ring, &ring.one(), try_generic_abs_square_and_multiply::<_, _, _, _, _, !>(
            &x, 
            &536903680, 
            StaticRing::<i64>::RING, 
            |mut res| {
                ring.square(&mut res);
                return Ok(res);
            }, 
            |a, b| Ok(ring.mul_ref_fst(a, b)), 
            ring.one()
        ).unwrap());
    });
}

#[bench]
fn bench_addchain_square_and_multiply(bencher: &mut Bencher) {
    let ring = zn_64::Zn::new(536903681);
    let x = ring.int_hom().map(2);
    bencher.iter(|| {
        assert_el_eq!(&ring, &ring.one(), generic_pow_shortest_chain_table::<_, _, _, _, _, !>(
            x, 
            &536903680, 
            StaticRing::<i64>::RING, 
            |a| {
                let mut res = ring.clone_el(a);
                ring.square(&mut res);
                return Ok(res);
            }, 
            |a, b| Ok(ring.mul_ref(a, b)), 
            |a| ring.clone_el(a),
            ring.one()
        ).unwrap());
    });
}