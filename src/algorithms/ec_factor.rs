use crate::algorithms;
use crate::ordered::OrderedRingWrapper;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::integer::*;
use crate::rings::zn::ZnRingWrapper;
use crate::rings::zn::zn_dyn::Zn;
use crate::rings::zn::zn_dyn::ZnBase;

#[allow(non_snake_case)]
fn ec_group_action<I>(Zn: &Zn<I>, A: &El<Zn<I>>, B: &El<Zn<I>>, P: Option<(El<Zn<I>>, El<Zn<I>>)>, Q: Option<(El<Zn<I>>, El<Zn<I>>)>) -> Result<Option<(El<Zn<I>>, El<Zn<I>>)>, El<I>>
    where I: IntegerRingWrapper
{
    let half = Zn.get_ring().invert(Zn.from_z(2)).ok().unwrap();
    match (P, Q) {
        (None, Q) => Ok(Q),
        (P, None) => Ok(P),
        (Some((P_x, P_y)), Some((Q_x, Q_y))) => {
            if Zn.eq(&P_x, &Q_x) && Zn.eq(&P_y, &Zn.negate(Q_y.clone())) {
                Ok(None)
            } else if Zn.eq(&P_x, &Q_x) {
                let s = Zn.mul(
                    Zn.add_ref_snd(
                        Zn.mul(Zn.from_z(3), Zn.mul_ref(&P_x, &P_x)), 
                        A
                    ),
                    Zn.mul_ref_fst(&half, Zn.get_ring().invert(P_y)?)
                );
                let R_x = Zn.sub(
                    Zn.mul_ref(&s, &s),
                    Zn.mul_ref_snd(Zn.from_z(2), &P_x)
                );
                let R_y = Zn.add(Q_y, Zn.mul(s, Zn.sub_ref_fst(&R_x, P_x)));
                Ok(Some((R_x, R_y)))
            } else {
                let s = Zn.mul(
                    Zn.sub_ref_fst(&P_y, Q_y),
                    Zn.get_ring().invert(Zn.sub_ref_fst(&P_x, Q_x))?
                );
                let R_x = Zn.sub(
                    Zn.mul_ref(&s, &s),
                    Zn.mul_ref_snd(Zn.from_z(2), &P_x)
                );
                let R_y = Zn.add(P_y, Zn.mul(s, Zn.sub_ref_fst(&R_x, P_x)));
                Ok(Some((R_x, R_y)))
            }
        }
    }
}

///
/// Runtime L_N(1/2, 1) = exp((1 + o(1)) ln(N)^1/2 lnln(N)^1/2)
/// 
#[allow(non_snake_case)]
fn lenstra_ec_factor<I>(ZZ: I, N: &El<I>) -> El<I>
    where I: IntegerRingWrapper
{
    assert!(ZZ.is_geq(N, &ZZ.from_z(100)));
    let Nf = ZZ.to_float_approx(N);
    // smoothness bound, choose L_N(1/2, 1/2)
    let B = (0.5 * Nf.ln().sqrt() * Nf.ln().ln().sqrt()).exp() as usize;
    let primes = algorithms::primes::enumerate_primes(&ZZ, &ZZ.map_in::<StaticRing<i128>>(&StaticRing::<i128>::RING, B as i128));
    let k = ZZ.prod(
        primes.iter()
            .map(|p| (Nf.log2() / ZZ.to_float_approx(&p).log2(), p))
            .map(|(e, p)| ZZ.pow(p, e as usize))
    );
    let Zn = Zn::new(ZnBase::new(&ZZ, N.clone()));
    let mut rng = oorandom::Rand64::new(ZZ.default_hash(N) as u128);
    loop {
        let P = (Zn.random_element(|| rng.rand_u64()), Zn.random_element(|| rng.rand_u64()));
        let A = Zn.random_element(|| rng.rand_u64());
        let B = Zn.sub(Zn.mul_ref(&P.1, &P.1), Zn.add(Zn.pow(&P.0, 3), Zn.mul_ref(&A, &P.0)));
        let result = algorithms::sqr_mul::generic_abs_square_and_multiply(
            &Ok(Some(P)),
            &k,
            &ZZ,
            |a, b| a.and_then(|a| b.and_then(|b| ec_group_action(&Zn, &A, &B, a, b))),
            |a, b| a.clone().and_then(|a| b.clone().and_then(|b| ec_group_action(&Zn, &A, &B, a, b))),
            Ok(None),
        );
        if let Err(factor) = result {
            return factor;
        }
    }
}