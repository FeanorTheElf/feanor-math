use crate::algorithms;
use crate::divisibility::*;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::integer::*;
use crate::rings::zn::ZnRing;
use crate::rings::zn::ZnRingStore;

#[allow(type_alias_bounds)]
type Point<R> = (El<R>, El<R>, El<R>);

fn ec_group_add_proj<R>(Zn: &R, _A: &El<R>, _B: &El<R>, P: Point<R>, Q: &Point<R>) -> Point<R> 
    where R: ZnRingStore,
        R::Type: ZnRing
{
    if Zn.is_zero(&Q.2) {
        return P;
    } else if Zn.is_zero(&P.2) {
        return (Zn.clone_el(&Q.0), Zn.clone_el(&Q.1), Zn.clone_el(&Q.2));
    }

    let (P_x, P_y, P_z) = P;
    let (Q_x, Q_y, Q_z) = Q;

    let u = Zn.sub(Zn.mul_ref(&P_y, &Q_z), Zn.mul_ref(&Q_y, &P_z));
    let w = Zn.sub(Zn.mul_ref(&P_x, &Q_z), Zn.mul_ref(&Q_x, &P_z));
    let w_sqr = Zn.mul_ref(&w, &w);
    let w_cbe = Zn.mul_ref(&w_sqr, &w);

    let R_x_div_w = Zn.sub(
        Zn.mul(Zn.mul_ref(&P_z, &Q_z), Zn.mul_ref(&u, &u)),
        Zn.mul_ref_snd(
            Zn.add(Zn.mul_ref(&P_x, &Q_z), Zn.mul_ref(&P_z, &Q_x)),
            &w_sqr, 
        )
    );
    return (
        Zn.mul_ref(&w, &R_x_div_w),
        Zn.add(
            Zn.mul(P_y, Zn.mul_ref(&Q_z, &w_cbe)),
            Zn.mul(u, Zn.sub(
                R_x_div_w, Zn.mul(P_x, Zn.mul_ref_fst(&Q_z, w_sqr))
            ))
        ),
        Zn.mul(P_z, w_cbe)
    );
}

fn ec_group_double_proj<R>(Zn: &R, A: &El<R>, _B: &El<R>, P: &Point<R>) -> Point<R>
    where R: ZnRingStore,
        R::Type: ZnRing
{
    let (x, y, z) = P;

    if Zn.is_zero(&z) {
        return (Zn.zero(), Zn.one(), Zn.zero());
    }
    
    let x_sqr = Zn.mul_ref(&x, &x);
    let x_sqr_3 = Zn.add(Zn.add_ref(&x_sqr, &x_sqr), x_sqr);
    let u = Zn.add(x_sqr_3, Zn.mul_ref_fst(&A, Zn.mul_ref(&z, &z)));
    let w = Zn.mul_ref(&y, &z);
    let w_sqr = Zn.mul_ref(&w, &w);
    let w_cbe = Zn.mul_ref(&w_sqr, &w);

    let x_w_sqr = Zn.mul_ref(&x, &w_sqr);
    let R_x_div_w = Zn.sub(
        Zn.mul_ref_fst(&z, Zn.mul_ref(&u, &u)),
        Zn.add_ref(&x_w_sqr, &x_w_sqr)
    );

    return (
        Zn.mul_ref(&w, &R_x_div_w),
        Zn.add(
            Zn.mul_ref(y, &w_cbe),
            Zn.mul(u, Zn.sub(R_x_div_w, x_w_sqr))
        ),
        Zn.mul_ref_fst(z, w_cbe)
    );
}

fn ec_pow_prime_abort<R>(base: &Point<R>, A: &El<R>, B: &El<R>, power: &i128, Zn: &R) -> Result<Point<R>, Point<R>>
    where R: ZnRingStore,
        R::Type: ZnRing
{
    let ZZ = StaticRing::<i128>::RING;
    if ZZ.is_zero(&power) {
        return Ok((Zn.zero(), Zn.one(), Zn.zero()));
    } else if ZZ.is_one(&power) {
        return Ok((Zn.clone_el(&base.0), Zn.clone_el(&base.1), Zn.clone_el(&base.2)));
    }

    let mut result = (Zn.zero(), Zn.one(), Zn.zero());
    for i in (0..=ZZ.abs_highest_set_bit(power).unwrap()).rev() {
        let double_result = ec_group_double_proj(Zn, A, B, &result);
        let new = if ZZ.abs_is_bit_set(power, i) {
            ec_group_add_proj(Zn, A, B, double_result, &base)
        } else {
            double_result
        };
        if Zn.is_zero(&new.2) {
            return Err(result);
        }
        result = new;
    }
    return Ok(result);
}

fn ec_pow_abort<R>(base: Point<R>, A: &El<R>, B: &El<R>, power_factorization: &[(i128, usize)], Zn: &R) -> Result<Point<R>, Point<R>>
    where R: ZnRingStore,
        R::Type: ZnRing
{
    let mut current = base;
    for (p, e) in power_factorization {
        for _ in 0..*e {
            current = ec_pow_prime_abort(&current, A, B, p, Zn)?;
        }
    }
    return Ok(current);
}

fn is_on_curve<R>(Zn: &R, A: &El<R>, B: &El<R>, P: &Point<R>) -> bool
    where R: ZnRingStore,
        R::Type: ZnRing
{
    let (x, y, z) = &P;
    Zn.eq_el(
        &Zn.mul_ref_snd(Zn.mul_ref(y, y), &z),
        &Zn.add(
            Zn.pow(Zn.clone_el(x), 3), Zn.mul(
                Zn.add(Zn.mul_ref(A, x), Zn.mul_ref(B, &z)),
                Zn.mul_ref(&z, &z)
            )
        )
    )
}

///
/// Optimizes the parameters to find a factor of size roughly size; size should be at most sqrt(N)
/// 
fn lenstra_ec_factor_base<R>(Zn: R, log2_size: usize, rng: &mut oorandom::Rand64) -> Option<El<<R::Type as ZnRing>::Integers>>
    where R: ZnRingStore + DivisibilityRingStore + Copy,
        R::Type: ZnRing + DivisibilityRing
{
    let ZZ = BigIntRing::RING;
    assert!(ZZ.is_leq(&ZZ.power_of_two(log2_size * 2), &Zn.size(&ZZ).unwrap()));
    let log2_N = ZZ.abs_log2_ceil(&Zn.size(&ZZ).unwrap()).unwrap();
    let log2_B = (log2_size as f64 * 2f64.ln() * (log2_N as f64 * 2f64.ln()).ln()).sqrt() / 2f64.ln();
    assert!(log2_B <= i128::MAX as f64);

    let primes = algorithms::erathostenes::enumerate_primes(&StaticRing::<i128>::RING, &(1i128 << (log2_B as u64)));
    let power_factorization = primes.iter()
            .map(|p| (*p, log2_B.ceil() as usize / StaticRing::<i128>::RING.abs_log2_ceil(&p).unwrap()))
            .collect::<Vec<_>>();

    // after this many random curves, we expect to have found a factor with high probability, unless there is no factor of size about `log2_size`
    for _ in 0..(1i128 << (log2_B as u64)) {
        let P = (Zn.random_element(|| rng.rand_u64()), Zn.random_element(|| rng.rand_u64()), Zn.one());
        let A = Zn.random_element(|| rng.rand_u64());
        let B = Zn.sub(Zn.mul_ref(&P.1, &P.1), Zn.add(Zn.pow(Zn.clone_el(&P.0), 3), Zn.mul_ref(&A, &P.0)));
        debug_assert!(is_on_curve(&Zn, &A, &B, &P));
        let result = ec_pow_abort(P, &A, &B, &power_factorization, &Zn).unwrap_or_else(|point| point);
        let possible_factor = algorithms::eea::gcd(Zn.smallest_positive_lift(result.2), Zn.integer_ring().clone_el(Zn.modulus()), Zn.integer_ring());
        if !Zn.integer_ring().is_unit(&possible_factor) {
            return Some(possible_factor);
        }
    }
    return None;
}

#[stability::unstable(feature = "enable")]
pub fn lenstra_ec_factor<R>(Zn: R) -> El<<R::Type as ZnRing>::Integers>
    where R: ZnRingStore + DivisibilityRingStore,
        R::Type: ZnRing + DivisibilityRing
{
    assert!(algorithms::miller_rabin::is_prime_base(&Zn, 10) == false);
    let ZZ = BigIntRing::RING;
    let log2_N = ZZ.abs_log2_ceil(&Zn.size(&ZZ).unwrap()).unwrap();
    let mut rng = oorandom::Rand64::new(Zn.integer_ring().default_hash(Zn.modulus()) as u128);

    // we first try to find smaller factors
    for log2_size in (16..(log2_N / 2)).step_by(8) {
        if let Some(factor) = lenstra_ec_factor_base(&Zn, log2_size, &mut rng) {
            return factor;
        }
    }
    // this is now the general case
    loop {
        if let Some(factor) = lenstra_ec_factor_base(&Zn, log2_N / 2, &mut rng) {
            return factor;
        }
    }
}

#[cfg(test)]
use crate::rings::zn::zn_64::Zn;
#[cfg(test)]
use std::time::Instant;
#[cfg(test)]
use crate::rings::zn::zn_big;
#[cfg(test)]
use test::Bencher;

#[test]
fn test_ec_factor() {
    let n = 11 * 17;
    let actual = lenstra_ec_factor(&Zn::new(n as u64));
    assert!(actual != 1 && actual != n && n % actual == 0);
    
    let n = 23 * 59 * 113;
    let actual = lenstra_ec_factor(&Zn::new(n as u64));
    assert!(actual != 1 && actual != n && n % actual == 0);
}

#[bench]
fn bench_ec_factor(bencher: &mut Bencher) {
    let bits = 58;
    let n = (1 << bits) + 1;
    let ring = Zn::new((1 << bits) + 1);

    bencher.iter(|| {
        let p = lenstra_ec_factor(&ring);
        assert!(n > 0 && n != 1 && n != p);
        assert!(n % p == 0);
    });
}

#[test]
#[ignore]
fn test_ec_factor_large() {
    let n: i128 = 127540261 * 71316922984999;

    let begin = Instant::now();
    let p = StaticRing::<i128>::RING.coerce(&BigIntRing::RING, lenstra_ec_factor(&zn_big::Zn::new(BigIntRing::RING, BigIntRing::RING.coerce(&StaticRing::<i128>::RING, n))));
    let end = Instant::now();
    println!("Done in {} ms", (end - begin).as_millis());
    assert!(p == 127540261 || p == 71316922984999);
}