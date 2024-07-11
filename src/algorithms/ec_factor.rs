use crate::algorithms;
use crate::divisibility::*;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::integer::*;
use crate::rings::zn::ZnRing;
use crate::rings::zn::ZnRingStore;

type Point<R> = (El<R>, El<R>, El<R>);

fn square<R>(Zn: &R, x: &El<R>) -> El<R>
    where R: RingStore
{
    let mut result: <<R as RingStore>::Type as RingBase>::Element = Zn.clone_el(&x);
    Zn.square(&mut result);
    return result;
}

fn edcurve_add<R>(Zn: &R, d: &El<R>, P: Point<R>, Q: &Point<R>) -> Point<R> 
    where R: ZnRingStore,
        R::Type: ZnRing
{
    let (Px, Py, Pz) = P;
    let (Qx, Qy, Qz) = Q;

    let PxQx = Zn.mul_ref(&Px, Qx);
    let PyQy = Zn.mul_ref(&Py, Qy);
    let PzQz = Zn.mul_ref_snd(Pz, Qz);

    let PzQz_sqr = square(Zn, &PzQz);
    let dPxPyQxQy = Zn.mul_ref_snd(Zn.mul_ref(&PxQx, &PyQy), d);

    let u1 = Zn.add_ref(&PzQz_sqr, &dPxPyQxQy);
    let u2 = Zn.sub(PzQz_sqr, dPxPyQxQy);

    let result = (
        Zn.mul_ref_fst(&PzQz, Zn.mul_ref_snd(Zn.add(Zn.mul_ref_snd(Px, Qy), Zn.mul_ref_snd(Py, Qx)), &u2)),
        Zn.mul(PzQz, Zn.mul_ref_snd(Zn.sub(PyQy, PxQx), &u1)),
        Zn.mul(u1, u2),
    );
    debug_assert!(is_on_curve(Zn, d, &result));
    return result;
}

fn edcurve_double<R>(Zn: &R, d: &El<R>, P: &Point<R>) -> Point<R> 
    where R: ZnRingStore,
        R::Type: ZnRing
{
    let (Px, Py, Pz) = P;

    let PxPy = Zn.mul_ref(&Px, &Py);
    let Px_sqr = square(Zn, Px);
    let Py_sqr = square(Zn, Py);
    let Pz_sqr = square(Zn, Pz);
    let Pz_pow4 = square(Zn, &Pz_sqr);
    let d_PxPy_sqr = Zn.mul_ref_snd(Zn.mul_ref(&Px_sqr, &Py_sqr), d);

    let u1 = Zn.add_ref(&Pz_pow4, &d_PxPy_sqr);
    let u2 = Zn.sub(Pz_pow4, d_PxPy_sqr);

    let result = (
        Zn.mul_ref_fst(&Pz_sqr, Zn.mul_ref_snd(Zn.add_ref(&PxPy, &PxPy), &u2)),
        Zn.mul_ref_fst(&Pz_sqr, Zn.mul_ref_snd(Zn.sub(Py_sqr, Px_sqr), &u1)),
        Zn.mul(u1, u2),
    );
    debug_assert!(is_on_curve(Zn, d, &result));
    return result;
}

fn ec_pow_prime_abort<R>(base: &Point<R>, d: &El<R>, power: &i128, Zn: &R) -> Result<Point<R>, Point<R>>
    where R: ZnRingStore,
        R::Type: ZnRing
{
    let ZZ = StaticRing::<i128>::RING;
    if ZZ.is_zero(&power) {
        return Ok((Zn.zero(), Zn.one(), Zn.one()));
    } else if ZZ.is_one(&power) {
        return Ok((Zn.clone_el(&base.0), Zn.clone_el(&base.1), Zn.clone_el(&base.2)));
    }

    let mut result = (Zn.zero(), Zn.one(), Zn.one());
    for i in (0..=ZZ.abs_highest_set_bit(power).unwrap()).rev() {
        let double_result = edcurve_double(Zn, d, &result);
        let new = if ZZ.abs_is_bit_set(power, i) {
            edcurve_add(Zn, d, double_result, &base)
        } else {
            double_result
        };
        if Zn.is_zero(&new.0) && (!Zn.is_zero(&new.2) || Zn.is_zero(&new.1) ) {
            return Err(result);
        }
        result = new;
    }
    return Ok(result);
}

fn ec_pow_abort<R>(base: Point<R>, d: &El<R>, power_factorization: &[(i128, usize)], Zn: &R) -> Result<Point<R>, Point<R>>
    where R: ZnRingStore,
        R::Type: ZnRing
{
    let mut current = base;
    for (p, e) in power_factorization {
        for _ in 0..*e {
            current = ec_pow_prime_abort(&current, d, p, Zn)?;
        }
    }
    return Ok(current);
}

fn is_on_curve<R>(Zn: &R, d: &El<R>, P: &Point<R>) -> bool
    where R: ZnRingStore,
        R::Type: ZnRing
{
    let (x, y, z) = &P;
    let x_sqr = square(Zn, x);
    let y_sqr = square(Zn, y);
    let z_sqr = square(Zn, z);
    Zn.eq_el(
        &Zn.mul_ref_snd(Zn.add_ref(&x_sqr, &y_sqr), &z_sqr),
        &Zn.add(
            Zn.mul_ref(&z_sqr, &z_sqr),
            Zn.mul_ref_fst(d, Zn.mul(x_sqr, y_sqr))
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
        let (x, y) = (Zn.random_element(|| rng.rand_u64()), Zn.random_element(|| rng.rand_u64()));
        let (x_sqr, y_sqr) = (square(&Zn, &x), square(&Zn, &y));
        if let Some(d) = Zn.checked_div(&Zn.sub(Zn.add_ref(&x_sqr, &y_sqr), Zn.one()), &Zn.mul(x_sqr, y_sqr)) {
            let P = (x, y, Zn.one());
            debug_assert!(is_on_curve(&Zn, &d, &P));
            let result = ec_pow_abort(P, &d, &power_factorization, &Zn).unwrap_or_else(|point| point);
            let possible_factor = algorithms::eea::gcd(Zn.smallest_positive_lift(result.0), Zn.integer_ring().clone_el(Zn.modulus()), Zn.integer_ring());
            if !Zn.integer_ring().is_unit(&possible_factor) && !Zn.integer_ring().eq_el(&possible_factor, Zn.modulus()) {
                return Some(possible_factor);
            }
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
    let log2_N = ZZ.abs_log2_floor(&Zn.size(&ZZ).unwrap()).unwrap();
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
    let n = ((1i64 << bits) + 1) / 5;
    let ring = Zn::new(n as u64);

    bencher.iter(|| {
        let p = lenstra_ec_factor(&ring);
        assert!(n > 0 && n != 1 && n != p);
        assert!(n % p == 0);
    });
}

#[test]
#[ignore]
fn test_ec_factor_large() {
    #[cfg(not(feature = "mpir"))]
    let ZZbig = crate::rings::rust_bigint::RustBigintRing::new_with(feanor_mempool::AllocRc(std::rc::Rc::new(feanor_mempool::dynsize::DynLayoutMempool::<std::alloc::Global>::new(std::ptr::Alignment::of::<u64>()))));
    #[cfg(feature = "mpir")]
    let ZZbig = BigIntRing::RING;

    let n: i128 = 1073741827 * 71316922984999;

    let begin = Instant::now();
    let p = StaticRing::<i128>::RING.coerce(&ZZbig, lenstra_ec_factor(&zn_big::Zn::new(&ZZbig, ZZbig.coerce(&StaticRing::<i128>::RING, n))));
    let end = Instant::now();
    println!("Done in {} ms", (end - begin).as_millis());
    assert!(p == 1073741827 || p == 71316922984999);

    let n: i128 = 1152921504606847009 * 2305843009213693967;

    let begin = Instant::now();
    let p = StaticRing::<i128>::RING.coerce(&ZZbig, lenstra_ec_factor(&zn_big::Zn::new(&ZZbig, ZZbig.coerce(&StaticRing::<i128>::RING, n))));
    let end = Instant::now();
    println!("Done in {} ms", (end - begin).as_millis());
    assert!(p == 1152921504606847009 || p == 2305843009213693967);
}