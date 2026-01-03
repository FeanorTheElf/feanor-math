use std::sync::atomic::AtomicU64;

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use tracing::{Level, event, instrument};

use crate::algorithms;
use crate::divisibility::*;
use crate::homomorphism::Homomorphism;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::finite::*;
use crate::integer::*;
use crate::rings::zn::*;
use crate::pid::PrincipalIdealRingStore;
use crate::algorithms::sqr_mul;
use crate::MAX_PROBABILISTIC_REPETITIONS;
use super::int_factor::is_prime_power;

type Point<R> = (El<R>, El<R>, El<R>);

fn square<R>(Zn: &R, x: &El<R>) -> El<R>
    where R: RingStore
{
    let mut result: <<R as RingStore>::Type as RingBase>::Element = Zn.clone_el(&x);
    Zn.square(&mut result);
    return result;
}

#[allow(unused)]
fn point_eq<R>(Zn: &R, P: &Point<R>, Q: &Point<R>) -> bool
    where R: RingStore,
        R::Type: ZnRing
{
    let factor_quo = if !Zn.is_zero(&Q.0) {
        if Zn.is_zero(&P.0) { return false; }
        (&P.0, &Q.0)
    } else if !Zn.is_zero(&Q.1) {
        if Zn.is_zero(&P.1) { return false; }
        (&P.1, &Q.1)
    } else {
        assert!(!Zn.is_zero(&Q.2));
        if Zn.is_zero(&P.2) { return false; }
        (&P.2, &Q.2)
    };
    if !Zn.is_unit(&factor_quo.1) {
        let factor_of_n = Zn.integer_ring().ideal_gen(Zn.modulus(), &Zn.smallest_positive_lift(Zn.clone_el(&factor_quo.1)));
        let Zn_new = zn_big::ZnGB::new(BigIntRing::RING, int_cast(Zn.integer_ring().checked_div(Zn.modulus(), &factor_of_n).unwrap(), BigIntRing::RING, Zn.integer_ring()));
        let red_map = ZnReductionMap::new(Zn, &Zn_new).unwrap();
        if (Zn_new.is_zero(&red_map.map_ref(&Q.0)) && Zn_new.is_zero(&red_map.map_ref(&Q.1)) && Zn_new.is_zero(&red_map.map_ref(&Q.2))) || (Zn_new.is_zero(&red_map.map_ref(&P.0)) && Zn_new.is_zero(&red_map.map_ref(&P.1)) && Zn_new.is_zero(&red_map.map_ref(&P.2))) {
            if (Zn_new.is_zero(&red_map.map_ref(&P.0)) && Zn_new.is_zero(&red_map.map_ref(&P.1)) && Zn_new.is_zero(&red_map.map_ref(&P.2))) != (Zn_new.is_zero(&red_map.map_ref(&Q.0)) && Zn_new.is_zero(&red_map.map_ref(&Q.1)) && Zn_new.is_zero(&red_map.map_ref(&Q.2))) {
                return false;
            }
        } else if !point_eq(&Zn_new, &(red_map.map_ref(&P.0), red_map.map_ref(&P.1), red_map.map_ref(&P.2)), &(red_map.map_ref(&Q.0), red_map.map_ref(&Q.1), red_map.map_ref(&Q.2))) {
            return false;
        }

        let Zn_new = zn_big::ZnGB::new(BigIntRing::RING, int_cast(factor_of_n, BigIntRing::RING, Zn.integer_ring()));
        let red_map = ZnReductionMap::new(Zn, &Zn_new).unwrap();
        if (Zn_new.is_zero(&red_map.map_ref(&Q.0)) && Zn_new.is_zero(&red_map.map_ref(&Q.1)) && Zn_new.is_zero(&red_map.map_ref(&Q.2))) || (Zn_new.is_zero(&red_map.map_ref(&P.0)) && Zn_new.is_zero(&red_map.map_ref(&P.1)) && Zn_new.is_zero(&red_map.map_ref(&P.2))) {
            if (Zn_new.is_zero(&red_map.map_ref(&P.0)) && Zn_new.is_zero(&red_map.map_ref(&P.1)) && Zn_new.is_zero(&red_map.map_ref(&P.2))) != (Zn_new.is_zero(&red_map.map_ref(&Q.0)) && Zn_new.is_zero(&red_map.map_ref(&Q.1)) && Zn_new.is_zero(&red_map.map_ref(&Q.2))) {
                return false;
            }
        } else if !point_eq(&Zn_new, &(red_map.map_ref(&P.0), red_map.map_ref(&P.1), red_map.map_ref(&P.2)), &(red_map.map_ref(&Q.0), red_map.map_ref(&Q.1), red_map.map_ref(&Q.2))) {
            return false;
        }
        return true;
    }
    let factor = Zn.checked_div(&factor_quo.0, &factor_quo.1).unwrap();
    if !Zn.is_unit(&factor) {
        return false;
    }
    return Zn.eq_el(&P.0, &Zn.mul_ref(&factor, &Q.0)) && Zn.eq_el(&P.1, &Zn.mul_ref(&factor, &Q.1)) && Zn.eq_el(&P.2, &Zn.mul_ref(&factor, &Q.2));
}

#[inline(never)]
#[instrument(skip_all, level = "trace")]
fn edcurve_add<R>(Zn: &R, d: &El<R>, P: Point<R>, Q: &Point<R>) -> Point<R> 
    where R: RingStore,
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

#[inline(never)]
#[instrument(skip_all, level = "trace")]
fn edcurve_double<R>(Zn: &R, d: &El<R>, P: &Point<R>) -> Point<R> 
    where R: RingStore,
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

#[instrument(skip_all, level = "trace")]
fn ec_pow<R>(base: &Point<R>, d: &El<R>, power: &El<BigIntRing>, Zn: &R) -> Point<R>
    where R: RingStore,
        R::Type: ZnRing
{
    let copy_point = |(x, y, z): &Point<R>| (Zn.clone_el(x), Zn.clone_el(y), Zn.clone_el(z));
    let ZZ = BigIntRing::RING;

    sqr_mul::generic_pow_shortest_chain_table(
        copy_point(base), 
        power, 
        ZZ, 
        |P| Ok(edcurve_double(Zn, d, &P)), 
        |P, Q| Ok(edcurve_add(Zn, d, copy_point(Q), P)), 
        |P| copy_point(P), 
        (Zn.zero(), Zn.one(), Zn.one())
    ).unwrap_or_else(|x| x)
}

#[instrument(skip_all, level = "trace")]
fn is_on_curve<R>(Zn: &R, d: &El<R>, P: &Point<R>) -> bool
    where R: RingStore,
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

const POW_COST_CONSTANT: f64 = 0.1;

///
/// returns `(ln_B, ln_attempts)`
/// 
#[instrument(skip_all, level = "trace")]
fn optimize_parameters(ln_p: f64, ln_n: f64) -> (f64, f64) {
    let pow_cost_constant = POW_COST_CONSTANT;
    let ln_cost_per_attempt = |ln_B: f64| ln_B + ln_B.ln() + pow_cost_constant * ln_n.ln();
    let ln_cost_per_attempt_diff = |ln_B: f64| 1. + 1./ln_B;
    let ln_attempts = |ln_B: f64| {
        let u = ln_p / ln_B;
        u * (1. + 2f64.ln()) * u.ln() - u
    };
    let ln_attempts_diff = |ln_B: f64| {
        let u = ln_p / ln_B;
        let u_diff = -ln_p / (ln_B * ln_B);
        u_diff * (1. + 2f64.ln()) * u.ln() + u * (1. + 2f64.ln()) * u_diff/u - u_diff
    };
    let f = |ln_B: f64| ln_cost_per_attempt(ln_B) - ln_attempts(ln_B);
    let f_diff = |ln_B: f64| ln_cost_per_attempt_diff(ln_B) - ln_attempts_diff(ln_B);

    let mut ln_B = (ln_p * ln_p.ln()).sqrt();
    for _ in 0..10 {
        ln_B = ln_B - f(ln_B) / f_diff(ln_B);
    }
    return (ln_B, ln_attempts(ln_B));
}

///
/// Optimizes the parameters to find a factor of size roughly `p`; `p` should be at most sqrt(n)
/// 
#[instrument(skip_all, level = "trace")]
fn lenstra_ec_factor_base<R, F>(Zn: R, log2_p: usize, mut rng: F) -> Option<El<<R::Type as ZnRing>::IntegerRing>>
    where R: RingStore + Copy,
        R::Type: ZnRing + DivisibilityRing,
        F: FnMut() -> u64 + Send
{
    event!(Level::TRACE, log2_n = Zn.integer_ring().abs_log2_ceil(Zn.modulus()).unwrap(), log2_p = log2_p);

    let ZZ = BigIntRing::RING;
    assert!(ZZ.is_leq(&ZZ.power_of_two(log2_p * 2), &Zn.size(&ZZ).unwrap()));
    let log2_n = ZZ.abs_log2_ceil(&Zn.size(&ZZ).unwrap()).unwrap();
    let ln_p = log2_p as f64 * 2f64.ln();
    let (ln_B, ln_attempts) = optimize_parameters(ln_p, log2_n as f64 * 2f64.ln());
    // after this many random curves, we expect to have found a factor with high probability, unless there is no factor of size about `log2_size`
    let attempts = ln_attempts.exp() as usize;
    event!(Level::TRACE, check_curves = attempts);

    let log2_B = ln_B / 2f64.ln();
    assert!(log2_B <= i128::MAX as f64);

    let primes = algorithms::erathostenes::enumerate_primes(&StaticRing::<i128>::RING, &(1i128 << (log2_B as u64)));
    let power_factorization = primes.iter()
        .map(|p| (*p, log2_B.ceil() as usize / StaticRing::<i128>::RING.abs_log2_ceil(&p).unwrap()))
        .collect::<Vec<_>>();
    let power = ZZ.prod(power_factorization.iter().map(|(p, e)| ZZ.pow(ZZ.coerce::<StaticRing<i128>>(&StaticRing::<i128>::RING, *p), *e)));
    let power_ref = &power;

    let base_rng_value = rng();
    let rng_seed = AtomicU64::new(1);

    let result = (0..attempts).into_par_iter().map(|_| {
        let mut rng = oorandom::Rand64::new(((rng_seed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as u128) << 64) | base_rng_value as u128);
        let (x, y) = (Zn.random_element(|| rng.rand_u64()), Zn.random_element(|| rng.rand_u64()));
        let (x_sqr, y_sqr) = (square(&Zn, &x), square(&Zn, &y));
        if let Some(d) = Zn.checked_div(&Zn.sub(Zn.add_ref(&x_sqr, &y_sqr), Zn.one()), &Zn.mul(x_sqr, y_sqr)) {
            let P = (x, y, Zn.one());
            debug_assert!(is_on_curve(&Zn, &d, &P));
            let result = ec_pow(&P, &d, power_ref, &Zn).0;
            if !Zn.is_unit(&result) && !Zn.is_zero(&result) {
                return Some(result);
            }
        }
        
        return None;
    }).find_any(|res| res.is_some())?;

    if let Some(result) = result {
        return Some(Zn.integer_ring().ideal_gen(&Zn.smallest_positive_lift(result), Zn.modulus()));
    } else {
        event!(Level::TRACE, "failed");
        return None;
    }
}

///
/// Given `Z/nZ`, tries to find a factor of `n` of size at most `2^min_factor_bound_log2`.
/// If such a factor exists, the function is likely to successfully find it (with probability
/// `1 - c^repetitions`, for a constant `0 < c < 1`). Otherwise, it is likely to return `None`.
/// 
/// Note that the returned value can be any nontrivial factor of `n`, and does not have to
/// be bounded by `2^min_factor_bound_log2`. The function is more likely to find small factors,
/// but can, in rare cases, find other factors as well.
/// 
/// # Explanation of logging output
/// 
/// If the passed computation controller accepts logging, it will receive the following symbols:
///  - `c(m)` means that `m` random curves will be tried using the current parameters
///  - `.` means an elliptic curve was tried and did not yield a factor of `n`
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn lenstra_ec_factor_small<R>(Zn: R, min_factor_bound_log2: usize, repetitions: usize) -> Option<El<<R::Type as ZnRing>::IntegerRing>>
    where R: ZnRingStore + DivisibilityRingStore + Copy,
        R::Type: ZnRing + DivisibilityRing
{
    assert!(algorithms::miller_rabin::is_prime_base(&Zn, 10) == false);
    assert!(is_prime_power(Zn.integer_ring(), Zn.modulus()).is_none());
    let mut rng = oorandom::Rand64::new(Zn.integer_ring().default_hash(Zn.modulus()) as u128);

    for log2_size in (16..min_factor_bound_log2).step_by(8) {
        if let Some(factor) = lenstra_ec_factor_base(Zn, log2_size, || rng.rand_u64()) {
            return Some(factor);
        }
    }
    for _ in 0..repetitions {
        if let Some(factor) = lenstra_ec_factor_base(Zn, min_factor_bound_log2, || rng.rand_u64()) {
            return Some(factor);
        }
    }
    return None;
}

#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn lenstra_ec_factor<R>(Zn: R) -> El<<R::Type as ZnRing>::IntegerRing>
    where R: ZnRingStore + DivisibilityRingStore + Copy,
        R::Type: ZnRing + DivisibilityRing
{
    assert!(algorithms::miller_rabin::is_prime_base(&Zn, 10) == false);
    assert!(is_prime_power(Zn.integer_ring(), Zn.modulus()).is_none());
    let ZZ = BigIntRing::RING;
    let log2_N = ZZ.abs_log2_floor(&Zn.size(&ZZ).unwrap()).unwrap();
    let mut rng = oorandom::Rand64::new(Zn.integer_ring().default_hash(Zn.modulus()) as u128);

    // we first try to find smaller factors
    for log2_size in (16..(log2_N / 2)).step_by(8) {
        if let Some(factor) = lenstra_ec_factor_base(Zn, log2_size, || rng.rand_u64()) {
            return factor;
        }
    }
    // this is now the general case
    for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
        if let Some(factor) = lenstra_ec_factor_base(Zn, log2_N / 2, || rng.rand_u64()) {
            return factor;
        }
    }
    unreachable!()
}

#[cfg(test)]
use crate::rings::zn::zn_64b::Zn64B;
#[cfg(test)]
use test::Bencher;
#[cfg(test)]
use crate::rings::rust_bigint::*;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[test]
fn test_ec_factor() {
    LogAlgorithmSubscriber::init_test();
    let n = 65537 * 65539;
    let actual = lenstra_ec_factor(&Zn64B::new(n as u64));
    assert!(actual != 1 && actual != n && n % actual == 0);
}

#[bench]
fn bench_ec_factor_mersenne_number_58(bencher: &mut Bencher) {
    LogAlgorithmSubscriber::init_test();
    let bits = 58;
    let n = ((1i64 << bits) + 1) / 5;
    let ring = Zn64B::new(n as u64);

    bencher.iter(|| {
        let p = lenstra_ec_factor(&ring);
        assert!(n > 0 && n != 1 && n != p);
        assert!(n % p == 0);
    });
}

#[test]
#[ignore]
fn test_ec_factor_large() {
    LogAlgorithmSubscriber::init_test();
    let ZZbig = BigIntRing::RING;

    let n: i128 = 1073741827 * 71316922984999;

    let p = StaticRing::<i128>::RING.coerce(&ZZbig, lenstra_ec_factor(&zn_big::ZnGB::new(&ZZbig, ZZbig.coerce::<StaticRing<i128>>(&StaticRing::<i128>::RING, n))));
    assert!(p == 1073741827 || p == 71316922984999);

    let n: i128 = 1152921504606847009 * 2305843009213693967;

    let p = StaticRing::<i128>::RING.coerce(&ZZbig, lenstra_ec_factor(&zn_big::ZnGB::new(&ZZbig, ZZbig.coerce::<StaticRing<i128>>(&StaticRing::<i128>::RING, n))));
    assert!(p == 1152921504606847009 || p == 2305843009213693967);
}

#[test]
#[ignore]
fn test_compute_partial_factorization() {
    LogAlgorithmSubscriber::init_test();
    let ZZbig = BigIntRing::RING;
    let n = int_cast(
        RustBigintRing::RING.get_ring().parse("5164499756173817179311838344006023748659411585658447025661318713081295244033682389259290706560275662871806343945494986751", 10).unwrap(),
        ZZbig, 
        RustBigintRing::RING
    );

    let Zn = zn_big::ZnGB::new(ZZbig, ZZbig.clone_el(&n));
    let factor = lenstra_ec_factor_small(&Zn, 50, 1).unwrap();
    ZZbig.println(&factor);
    assert!(!ZZbig.is_one(&factor));
    assert!(!ZZbig.eq_el(&factor, &n));
    assert!(ZZbig.divides(&n, &factor));
}
