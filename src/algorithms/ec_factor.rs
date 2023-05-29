use crate::algorithms;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::StaticRing;
use crate::primitive_int::StaticRingBase;
use crate::ring::*;
use crate::integer::*;
use crate::rings::zn::ZnRingStore;
use crate::rings::zn::zn_barett::Zn;

#[allow(type_alias_bounds)]
type Point<I: IntegerRingStore> = (El<Zn<I>>, El<Zn<I>>, El<Zn<I>>);

fn ec_group_action_proj<I>(Zn: &Zn<I>, _A: &El<Zn<I>>, _B: &El<Zn<I>>, P: Point<I>, Q: &Point<I>) -> Point<I> 
    where I: IntegerRingStore,
        I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
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

fn ec_group_double_proj<I>(Zn: &Zn<I>, A: &El<Zn<I>>, _B: &El<Zn<I>>, P: &Point<I>) -> Point<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
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

pub fn ec_mul_abort<I>(base: &Point<I>, A: &El<Zn<I>>, B: &El<Zn<I>>, power: &El<I>, ZZ: &I, Zn: &Zn<I>) -> Point<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    if ZZ.is_zero(&power) {
        return (Zn.zero(), Zn.one(), Zn.zero());
    } else if ZZ.is_one(&power) {
        return (Zn.clone_el(&base.0), Zn.clone_el(&base.1), Zn.clone_el(&base.2));
    }

    let mut result = (Zn.zero(), Zn.one(), Zn.zero());
    for i in (0..=ZZ.abs_highest_set_bit(power).unwrap()).rev() {
        let double_result = ec_group_double_proj(Zn, A, B, &result);
        let new = if ZZ.abs_is_bit_set(power, i) {
            ec_group_action_proj(Zn, A, B, double_result, &base)
        } else {
            double_result
        };
        if Zn.is_zero(&new.2) {
            return result;
        }
        result = new;
    }
    return result;
}

fn is_on_curve<I>(Zn: &Zn<I>, A: &El<Zn<I>>, B: &El<Zn<I>>, P: &Point<I>) -> bool
    where I: IntegerRingStore,
        I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
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
/// Runtime `L_N(1/2, 1) = exp((1 + o(1)) ln(N)^1/2 lnln(N)^1/2)`
/// 
pub fn lenstra_ec_factor<I>(ZZ: I, N: &El<I>) -> El<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing + CanonicalIso<StaticRingBase<i128>> + CanonicalIso<StaticRingBase<i32>>
{
    assert!(algorithms::miller_rabin::is_prime(&ZZ, N, 6) == false);
    assert!(ZZ.is_geq(N, &ZZ.from_int(100)));
    let Nf = ZZ.to_float_approx(N);
    // smoothness bound, choose L_N(1/2, 1/2)
    let B = (0.5 * Nf.ln().sqrt() * Nf.ln().ln().sqrt()).exp() as usize;
    let primes = algorithms::erathostenes::enumerate_primes(&ZZ, &ZZ.coerce::<StaticRing<i128>>(&StaticRing::<i128>::RING, B as i128));
    let k = ZZ.prod(
        primes.iter()
            .map(|p| (Nf.log2() / ZZ.to_float_approx(&p).log2(), p))
            .map(|(e, p)| ZZ.pow(ZZ.clone_el(p), e as usize + 1))
    );
    let Zn = Zn::new(&ZZ, ZZ.clone_el(&N));
    let mut rng = oorandom::Rand64::new(ZZ.default_hash(N) as u128);
    loop {
        let P = (Zn.random_element(|| rng.rand_u64()), Zn.random_element(|| rng.rand_u64()), Zn.one());
        let A = Zn.random_element(|| rng.rand_u64());
        let B = Zn.sub(Zn.mul_ref(&P.1, &P.1), Zn.add(Zn.pow(Zn.clone_el(&P.0), 3), Zn.mul_ref(&A, &P.0)));
        debug_assert!(is_on_curve(&Zn, &A, &B, &P));
        let result = ec_mul_abort(&P, &A, &B, &k, &&ZZ, &Zn);
        if let Err(factor) = Zn.get_ring().invert(result.2) {
            return factor;
        }
    }
}

#[cfg(test)]
use crate::divisibility::DivisibilityRingStore;
#[cfg(test)]
use crate::rings::bigint::DefaultBigIntRing;

#[test]
fn test_ec_factor() {
    let n = 11 * 17;
    let actual = lenstra_ec_factor(StaticRing::<i64>::RING, &n);
    assert!(actual != 1 && actual != n && n % actual == 0);
    
    let n = 23 * 59 * 113;
    let actual = lenstra_ec_factor(StaticRing::<i128>::RING, &n);
    assert!(actual != 1 && actual != n && n % actual == 0);
}

#[bench]
fn bench_ec_factor(bencher: &mut test::Bencher) {
    let ZZ = DefaultBigIntRing::RING;
    let mut n = ZZ.one();
    ZZ.mul_pow_2(&mut n, 48);
    ZZ.add_assign(&mut n, ZZ.one());
    bencher.iter(|| {
        let p = lenstra_ec_factor(ZZ, &n);
        assert!(ZZ.checked_div(&n, &p).is_some());
    });
}