use crate::divisibility::{DivisibilityRingStore, DivisibilityRing};
use crate::homomorphism::Homomorphism;
use crate::local::{PrincipalLocalRing, PrincipalLocalRingStore};
use crate::pid::PrincipalIdealRingStore;
use crate::ring::*;
use crate::rings::multivariate_new::*;

use std::fmt::Debug;
use std::io::Write;

const EXTENSIVE_LOG: bool = false;

#[derive(PartialEq, Clone, Eq, Hash)]
enum SPoly {
    Standard(usize, usize), Nilpotent(/* poly index */ usize, /* power-of-p multiplier */ usize)
}

impl Debug for SPoly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SPoly::Standard(i, j) => write!(f, "S({}, {})", i, j),
            SPoly::Nilpotent(i, k) => write!(f, "p^{} F({})", k, i)
        }
    }
}

fn term_xlcm<P>(ring: P, (l_c, l_m): (&PolyCoeff<P>, &PolyMonomial<P>), (r_c, r_m): (&PolyCoeff<P>, &PolyMonomial<P>)) -> ((PolyCoeff<P>, PolyMonomial<P>), (PolyCoeff<P>, PolyMonomial<P>), (PolyCoeff<P>, PolyMonomial<P>))
    where P: RingStore,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing
{
    let d_c = ring.base_ring().ideal_gen(l_c, r_c);
    let m_m = ring.monomial_lcm(ring.clone_monomial(l_m), r_m);
    let l_factor = ring.base_ring().checked_div(r_c, &d_c).unwrap();
    let r_factor = ring.base_ring().checked_div(l_c, &d_c).unwrap();
    let m_c = ring.base_ring().mul_ref_snd(ring.base_ring().mul_ref_snd(d_c, &r_factor), &l_factor);
    return (
        (l_factor, ring.monomial_div(ring.clone_monomial(&m_m), &l_m).ok().unwrap()),
        (r_factor, ring.monomial_div(ring.clone_monomial(&m_m), &r_m).ok().unwrap()),
        (m_c, m_m)
    );
}

fn term_lcm<P>(ring: P, (l_c, l_m): (&PolyCoeff<P>, &PolyMonomial<P>), (r_c, r_m): (&PolyCoeff<P>, &PolyMonomial<P>)) -> (PolyCoeff<P>, PolyMonomial<P>)
    where P: RingStore,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing
{
    let d_c = ring.base_ring().ideal_gen(l_c, r_c);
    let m_m = ring.monomial_lcm(ring.clone_monomial(l_m), r_m);
    let l_factor = ring.base_ring().checked_div(r_c, &d_c).unwrap();
    let r_factor = ring.base_ring().checked_div(l_c, &d_c).unwrap();
    let m_c = ring.base_ring().mul_ref_snd(ring.base_ring().mul_ref_snd(d_c, &r_factor), &l_factor);
    return (m_c, m_m);
}

impl SPoly {

    fn lcm_term<P, O>(&self, ring: P, basis: &[El<P>], order: O) -> (PolyCoeff<P>, PolyMonomial<P>)
        where P: RingStore + Copy,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
            O: MonomialOrder
    {
        match self {
            SPoly::Standard(i, j) => term_lcm(&ring, ring.LT(&basis[*i], order.clone()).unwrap(), ring.LT(&basis[*j], order.clone()).unwrap()),
            Self::Nilpotent(i, k) => {
                let (c, m) = ring.LT(&basis[*i], order.clone()).unwrap();
                (ring.base_ring().mul_ref_fst(c, ring.base_ring().pow(ring.base_ring().clone_el(ring.base_ring().max_ideal_gen()), *k)), ring.clone_monomial(m))
            }
        }
    }

    fn poly<P, O>(&self, ring: P, basis: &[El<P>], order: O) -> El<P>
        where P: RingStore + Copy,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
            O: MonomialOrder
    {
        match self {
            SPoly::Standard(i, j) => {
                let (f1_factor, f2_factor, _) = term_xlcm(&ring, ring.LT(&basis[*i], order.clone()).unwrap(), ring.LT(&basis[*j], order.clone()).unwrap());
                let mut f1_scaled = ring.clone_el(&basis[*i]);
                ring.mul_assign_monomial(&mut f1_scaled, f1_factor.1);
                ring.inclusion().mul_assign_map(&mut f1_scaled, f1_factor.0);
                let mut f2_scaled = ring.clone_el(&basis[*j]);
                ring.mul_assign_monomial(&mut f2_scaled, f2_factor.1);
                ring.inclusion().mul_assign_map(&mut f2_scaled, f2_factor.0);
                return ring.sub(f1_scaled, f2_scaled);
            },
            SPoly::Nilpotent(i, k) => {
                let mut result = ring.clone_el(&basis[*i]);
                ring.inclusion().mul_assign_map(&mut result, ring.base_ring().pow(ring.base_ring().clone_el(ring.base_ring().max_ideal_gen()), *k));
                return result;
            }
        }
    }
}

#[inline(never)]
fn find_reducer<'a, P, O, I>(ring: P, f: &El<P>, reducers: I, order: O) -> Option<(&'a El<P>, PolyCoeff<P>, PolyMonomial<P>)>
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
        O: MonomialOrder + Copy,
        I: Iterator<Item = &'a El<P>>
{
    if ring.is_zero(&f) {
        return None;
    }
    let (f_lc, f_lm) = ring.LT(f, order.clone()).unwrap();
    reducers.filter_map(|reducer| {
        let (r_lc, r_lm) = ring.LT(reducer, order.clone()).unwrap();
        if let Ok(quo_m) = ring.monomial_div(ring.clone_monomial(f_lm), r_lm) {
            if let Some(quo_c) = ring.base_ring().checked_div(f_lc, r_lc) {
                return Some((reducer, quo_c, quo_m));
            }
        }
        return None;
    }).next()
}

#[inline(never)]
fn filter_gebauer_moeller<P, O>(ring: P, new_spoly: SPoly, basis: &[El<P>], order: O) -> Option<usize>
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
        O: MonomialOrder
{
    match new_spoly {
        SPoly::Standard(i, k) => {
            assert!(i < k);
            let (bi_c, bi_m) = ring.LT(&basis[i], order.clone()).unwrap();
            let (bk_c, bk_m) = ring.LT(&basis[k], order.clone()).unwrap();
            let (S_c, S_m) = term_lcm(ring, (bi_c, bi_m), (bk_c, bk_m));
            let S_c_val = ring.base_ring().valuation(&S_c).unwrap();

            if S_c_val == 0 && order.eq_mon(ring, &ring.monomial_div(ring.clone_monomial(&S_m), &bi_m).ok().unwrap(), &bk_m) {
                return Some(usize::MAX);
            }

            (0..k).filter_map(|j| {
                if j == i {
                    return None;
                }
                let (bj_c, bj_m) = ring.LT(&basis[j], order.clone()).unwrap();
                let (f_c, f_m) = term_lcm(ring, (bj_c, bj_m), (bk_c, bk_m));
                let f_c_val = ring.base_ring().valuation(&f_c).unwrap();

                // TODO: can valuation inequality be relaxed?
                if j < i && order.eq_mon(ring, &f_m, &S_m) && S_c_val == f_c_val {
                    return Some(j);
                }
                if let Ok(quo) = ring.monomial_div(ring.clone_monomial(&S_m), &f_m) {
                    if ring.monomial_deg(&quo) > 0 && f_c_val <= S_c_val {
                        return Some(j);
                    }
                }
                return None;
            }).next()
        },
        SPoly::Nilpotent(i, k) => {
            let nilpotent_power = ring.base_ring().nilpotent_power().unwrap();
            let f = &basis[i];
            let mut last = ring.LT(f, order.clone()).unwrap();
            let mut current = if let Some(t) = ring.largest_term_lt(f, order.clone(), &last.1) {
                t
            } else {
                return Some(usize::MAX);
            };
            while ring.base_ring().valuation(&current.0).unwrap() + k >= nilpotent_power {
                let next = ring.largest_term_lt(f, order.clone(), &current.1);
                if next.is_none() {
                    return Some(usize::MAX);
                }
                last = current;
                current = next.unwrap();
            }
            if ring.base_ring().valuation(&last.0).unwrap() + k != nilpotent_power {
                return Some(usize::MAX);
            } else {
                return None;
            }
        }
    }
}

#[stability::unstable(feature = "enable")]
pub fn buchberger<P, O, const LOG: bool>(ring: P, basis: Vec<El<P>>, order: O) -> Vec<El<P>>
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: Sync,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
        O: MonomialOrder + Copy,
        PolyCoeff<P>: Send + Sync
{
    // this are the basis polynomials we generated; we only append to this, such that the S-polys
    // remain valid
    let mut basis = reduce(&ring, basis, order);
    assert!(basis.iter().all(|f| !ring.is_zero(f)));

    let nilpotent_power = ring.base_ring().nilpotent_power().and_then(|e| if e != 0 { Some(e) } else { None });

    if EXTENSIVE_LOG {
        println!("Basis:");
        for f in &basis {
            ring.println(f);
        }
    }

    let sort_open = |open: &mut [SPoly], basis: &[El<P>]| open.sort_unstable_by_key(|S_poly| -(ring.monomial_deg(&S_poly.lcm_term(ring, &basis, order.clone()).1) as i64));
    // the S-polys we have to consider
    let mut open = Vec::new();
    for i in 0..basis.len() {
        for j in (i + 1)..basis.len() {
            open.push(SPoly::Standard(i, j));
        }
    }
    if let Some(e) = nilpotent_power {
        for i in 0..basis.len() {
            for k in 1..e {
                open.push(SPoly::Nilpotent(i, k));
            }
        }
    }
    sort_open(&mut open, &basis);

    // this are the polynomials that we use to reduce S-polys; this is conceptually `reduce(basis)`
    let sort_reducers = |reducers: &mut [El<P>]| {
        reducers.sort_unstable_by(|lf, rf| order.compare(ring, &ring.LT(lf, order.clone()).unwrap().1, &ring.LT(rf, order.clone()).unwrap().1).then_with(|| ring.terms(lf).count().cmp(&ring.terms(rf).count())))
    };
    let mut reducers = basis.iter().map(|f| ring.clone_el(f)).collect::<Vec<_>>();
    sort_reducers(&mut reducers);

    let mut current_deg = 0;
    let mut new_polys = Vec::new();
    loop {

        while let Some(spoly) = open.last() {

            if let Some(filter_idx) = filter_gebauer_moeller(ring, spoly.clone(), &basis, order.clone()) {
                if EXTENSIVE_LOG {
                    if filter_idx == usize::MAX {
                        println!("Filter {:?} using product or nilpotent criterion", spoly);
                    } else {
                        println!("Filter {:?} using chain criterion and F({})", spoly, filter_idx);
                    }
                }
                open.pop();
                continue;
            } else {
                if EXTENSIVE_LOG {
                    println!("Consider {:?}", spoly);
                }
            }
    
            if ring.monomial_deg(&spoly.lcm_term(ring, &basis, order.clone()).1) > current_deg {
                break;
            }
            
            let spoly = open.pop().unwrap();
            let mut f = spoly.poly(ring, &basis, order.clone());
            
            while let Some((reducer, quo_c, quo_m)) = find_reducer(ring, &f, reducers.iter(), order.clone()) {
                ring.get_ring().add_assign_from_terms(&mut f, ring.terms(reducer).map(|(c, m)| (ring.base_ring().negate(ring.base_ring().mul_ref(c, &quo_c)), ring.monomial_mul(ring.clone_monomial(m), &quo_m))));
            }
            if !ring.is_zero(&f) {
                if EXTENSIVE_LOG {
                    println!("F({}) = ", basis.len());
                    ring.println(&f);
                }
                new_polys.push(ring.clone_el(&f));
                reducers.push(f);
                sort_reducers(&mut reducers);
            } else {
                if EXTENSIVE_LOG {
                    println!("reduced to zero");
                }
                if !EXTENSIVE_LOG && LOG {
                    print!(".");
                    std::io::stdout().flush().unwrap();
                }
            }
        }

        if new_polys.len() == 0 && open.len() == 0 {
            // as opposed to my initial belief, `basis` does not have to be a GB here
            return reduce(ring, reducers, order);

        } else if new_polys.len() == 0 {
            current_deg = ring.monomial_deg(&open.last().unwrap().lcm_term(ring, &basis, order.clone()).1);
            if !EXTENSIVE_LOG && LOG {
                print!("{{{}}}", current_deg);
                std::io::stdout().flush().unwrap();
            }
        } else {
            current_deg = 0;
            for new_poly in new_polys.drain(..) {
                basis.push(new_poly);
                for i in 0..(basis.len() - 1) {
                    let spoly = SPoly::Standard(i, basis.len() - 1);
                    open.push(spoly);
                }
                if let Some(e) = nilpotent_power {
                    for k in 1..e {
                        open.push(SPoly::Nilpotent(basis.len() - 1, k));
                    }
                }
            }
            sort_open(&mut open, &basis);
            if !EXTENSIVE_LOG && LOG {
                print!("b({})S({})", basis.len(), open.len());
                std::io::stdout().flush().unwrap();
            }
            reducers = reduce(ring, reducers, order);
            sort_reducers(&mut reducers);
            if !EXTENSIVE_LOG && LOG {
                print!("r({})", reducers.len());
                std::io::stdout().flush().unwrap();
            }
        }

        // less S-polys if we restart from scratch with reducers
        if open.len() > reducers.len() * reducers.len() / 2 + reducers.len() * nilpotent_power.unwrap_or(0) + 1 {
            if !EXTENSIVE_LOG && LOG {
                print!("!");
                std::io::stdout().flush().unwrap();
            }
            return buchberger::<P, O, LOG>(ring, reducers, order);
        }
    }
}

#[inline(never)]
#[stability::unstable(feature = "enable")]
pub fn reduce<P, O>(ring: P, mut polys: Vec<El<P>>, order: O) -> Vec<El<P>>
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: DivisibilityRingStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
        O: MonomialOrder + Copy
{
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i < polys.len() {
            let last_i = polys.len() - 1;
            polys.swap(i, last_i);
            let (reducers, f) = polys.split_at_mut(last_i);

            while let Some((reducer, quo_c, quo_m)) = find_reducer(ring, &f[0], reducers.iter(), order.clone()) {
                changed = true;
                ring.get_ring().add_assign_from_terms(&mut f[0], ring.terms(reducer).map(|(c, m)| (ring.base_ring().negate(ring.base_ring().mul_ref(c, &quo_c)), ring.monomial_mul(ring.clone_monomial(m), &quo_m))));
                if ring.is_zero(&f[0]) {
                    polys.truncate(last_i);
                    break;
                }
            }

            // undo swap so that the outer loop still iterates over every poly
            if polys.len() > last_i {
                polys.swap(i, last_i);
                i += 1;
            }
        }
    }
    return polys;
}

#[stability::unstable(feature = "enable")]
pub fn multivariate_division<'a, P, O, I>(ring: P, mut f: El<P>, reducers: I, order: O) -> El<P>
    where P: 'a + RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: DivisibilityRingStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
        O: MonomialOrder + Copy,
        I: Clone + Iterator<Item = &'a El<P>>
{
    while let Some((reducer, quo_c, quo_m)) = find_reducer(ring, &f, reducers.clone(), order.clone()) {
        ring.get_ring().add_assign_from_terms(&mut f, ring.terms(reducer).map(|(c, m)| (ring.base_ring().negate(ring.base_ring().mul_ref(c, &quo_c)), ring.monomial_mul(ring.clone_monomial(m), &quo_m))));
    }
    return f;
}

#[cfg(test)]
use crate::rings::multivariate_new::multivariate_impl::MultivariatePolyRingImpl;
#[cfg(test)]
use crate::rings::poly::{dense_poly, PolyRingStore};
#[cfg(test)]
use crate::rings::zn::zn_static;
#[cfg(test)]
use crate::rings::local::AsLocalPIR;

#[test]
fn test_buchberger_small() {
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 2);

    let f1 = ring.from_terms([
        (1, ring.create_monomial([2, 0])),
        (1, ring.create_monomial([0, 2])),
        (16, ring.create_monomial([0, 0]))
    ].into_iter());
    let f2 = ring.from_terms([
        (1, ring.create_monomial([1, 1])),
        (15, ring.create_monomial([0, 0]))
    ].into_iter());

    let actual = buchberger::<_, _, true>(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2)], DegRevLex);

    let expected = ring.from_terms([
        (16, ring.create_monomial([0, 3])),
        (15, ring.create_monomial([1, 0])),
        (1, ring.create_monomial([0, 1])),
    ].into_iter());

    assert_eq!(3, actual.len());
    assert_el_eq!(ring, ring.zero(), multivariate_division(&ring, f1, actual.iter(), DegRevLex));
    assert_el_eq!(ring, ring.zero(), multivariate_division(&ring, f2, actual.iter(), DegRevLex));
    assert_el_eq!(ring, ring.zero(), multivariate_division(&ring, expected, actual.iter(), DegRevLex));
}

#[test]
fn test_buchberger_larger() {
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 3);

    let f1 = ring.from_terms([
        (1, ring.create_monomial([2, 1, 1])),
        (1, ring.create_monomial([0, 2, 0])),
        (1, ring.create_monomial([1, 0, 1])),
        (2, ring.create_monomial([1, 0, 0])),
        (1, ring.create_monomial([0, 0, 0]))
    ].into_iter());
    let f2 = ring.from_terms([
        (1, ring.create_monomial([0, 3, 1])),
        (1, ring.create_monomial([0, 0, 3])),
        (1, ring.create_monomial([1, 1, 0]))
    ].into_iter());
    let f3 = ring.from_terms([
        (1, ring.create_monomial([1, 0, 2])),
        (1, ring.create_monomial([1, 0, 1])),
        (2, ring.create_monomial([0, 1, 1])),
        (7, ring.create_monomial([0, 0, 0]))
    ].into_iter());

    let actual = buchberger::<_, _, true>(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2), ring.clone_el(&f3)], DegRevLex);

    let g1 = ring.from_terms([
        (1, ring.create_monomial([0, 4, 0])),
        (8, ring.create_monomial([0, 3, 1])),
        (12, ring.create_monomial([0, 1, 3])),
        (6, ring.create_monomial([0, 0, 4])),
        (1, ring.create_monomial([0, 3, 0])),
        (13, ring.create_monomial([0, 2, 1])),
        (11, ring.create_monomial([0, 1, 2])),
        (10, ring.create_monomial([0, 0, 3])),
        (11, ring.create_monomial([0, 2, 0])),
        (12, ring.create_monomial([0, 1, 1])),
        (6, ring.create_monomial([0, 0, 2])),
        (6, ring.create_monomial([0, 1, 0])),
        (13, ring.create_monomial([0, 0, 1])),
        (9, ring.create_monomial([0, 0, 0]))
    ].into_iter());

    assert_el_eq!(ring, ring.zero(), multivariate_division(&ring, g1, actual.iter(), DegRevLex));
}

#[test]
fn test_generic_computation() {
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 6);
    let poly_ring = dense_poly::DensePolyRing::new(&ring, "X");

    let var_i = |i: usize| ring.create_term(base.one(), ring.create_monomial((0..ring.variable_count()).map(|j| if i == j { 1 } else { 0 })));
    let X1 = poly_ring.mul(
        poly_ring.from_terms([(var_i(0), 0), (ring.one(), 1)].into_iter()),
        poly_ring.from_terms([(var_i(1), 0), (ring.one(), 1)].into_iter())
    );
    let X2 = poly_ring.mul(
        poly_ring.add(poly_ring.clone_el(&X1), poly_ring.from_terms([(var_i(2), 0), (var_i(3), 1)].into_iter())),
        poly_ring.add(poly_ring.clone_el(&X1), poly_ring.from_terms([(var_i(4), 0), (var_i(5), 1)].into_iter()))
    );
    let basis = vec![
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 0)),
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 1)),
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 2)),
    ];

    let start = std::time::Instant::now();
    let gb1 = buchberger::<_, _, true>(&ring, basis.iter().map(|f| ring.clone_el(f)).collect(), DegRevLex);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    assert_eq!(11, gb1.len());
}

#[test]
fn test_gb_local_ring() {
    let base = AsLocalPIR::from_zn(zn_static::Zn::<16>::RING).unwrap();
    let ring: MultivariatePolyRingImpl<_> = MultivariatePolyRingImpl::new(base, 1);
    
    let f = ring.from_terms([(base.int_hom().map(4), ring.create_monomial([1])), (base.one(), ring.create_monomial([0]))].into_iter());
    let gb = buchberger::<_, _, true>(&ring, vec![f], DegRevLex);

    for g in &gb {
        ring.println(g);
    }

    assert_eq!(1, gb.len());
    assert_el_eq!(ring, ring.one(), gb[0]);
}

#[ignore]
#[test]
fn test_expensive_gb_1() {
    let base = AsLocalPIR::from_zn(zn_static::Zn::<16>::RING).unwrap();
    let ring: MultivariatePolyRingImpl<_> = MultivariatePolyRingImpl::new(base, 12);

    let system = ring.with_wrapped_indeterminates(|[Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11]| [
        Y0 * Y1 * Y2 * Y3.pow_ref(2) * Y4.pow_ref(2) + 4 * Y0 * Y1 * Y2 * Y3 * Y4 * Y4 * Y8 + Y0 * Y1 * Y2 * Y5.pow_ref(2) * Y8.pow_ref(2) + Y0 * Y2 * Y3 * Y4 * Y6 + Y0 * Y1 * Y3 * Y4 * Y7 + Y0 * Y2 * Y5 * Y6 * Y8 + Y0 * Y1 * Y5 * Y7 * Y8 + Y0 * Y2 * Y3 * Y5 * Y10 + Y0 * Y1 * Y3 * Y5 * Y11 + Y0 * Y6 * Y7 + Y3 * Y5 * Y9 -  4,
        2 * Y0 * Y1 * Y2 * Y3.pow_ref(2) * Y4 * Y5 + 2 * Y0 * Y1 * Y2 * Y3 * Y5.pow_ref(2) * Y8 + Y0 * Y2 * Y3 * Y5 * Y6 + Y0 * Y1 * Y3 * Y5 * Y7 + 8,
        Y0 * Y1 * Y2 * Y3.pow_ref(2) * Y5.pow_ref(2) - 5
    ]);

    let part_of_result = ring.with_wrapped_indeterminates(|[_Y0, Y1, Y2, _Y3, _Y4, _Y5, Y6, Y7, _Y8, _Y9, _Y10, _Y11]| [
        4 * Y2.pow_ref(2) * Y6.pow_ref(2) -  4 * Y1.pow_ref(2) * Y7.pow_ref(2),
        8 * Y2 * Y6 + 8 * Y1 * Y7.clone()
    ]);

    let start = std::time::Instant::now();
    let gb = buchberger::<_, _, true>(&ring, system, DegRevLex);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    for f in &part_of_result {
        assert!(ring.is_zero(&multivariate_division(&ring, ring.clone_el(f), gb.iter(), DegRevLex)));
    }

    assert_eq!(93, gb.len());
}

#[test]
#[ignore]
fn test_expensive_gb_2() {
    let base = zn_static::Fp::<7>::RING;
    let ring = MultivariatePolyRingImpl::new(base, 7);

    let basis = ring.with_wrapped_indeterminates(|[X0, X1, X2, X3, X4, X5, X6]| [
        6 + 2 * X5 + 2 * X4 + X6 + 4 * X0 + 5 * X6 * X5 + X6 * X4 + 3 * X0 * X4 + 6 * X0 * X6 + 2 * X0 * X3 + X0 * X2 + 4 * X0 * X1 + 2 * X3 * X4 * X5 + 4 * X0 * X6 * X5 + 6 * X0 * X2 * X5 + 5 * X0 * X6 * X4 + 2 * X0 * X3 * X4 + 4 * X0 * X1 * X4 + X0 * X6.pow_ref(2) + 3 * X0 * X3 * X6 + 5 * X0 * X2 * X6 + 2 * X0 * X1 * X6 + X0 * X3.pow_ref(2) + 2 * X0 * X2 * X3 + 3 * X0 * X3 * X4 * X5 + 4 * X0 * X3 * X6 * X5 + 3 * X0 * X1 * X6 * X5 + 3 * X0 * X2 * X3 * X5 + 3 * X0 * X3 * X6 * X4 + 2 * X0 * X1 * X6 * X4 + 2 * X0 * X3.pow_ref(2) * X4 + 2 * X0 * X2 * X3 * X4 + 3 * X0 * X3.pow_ref(2) * X4 * X5 + 4 * X0 * X1 * X3 * X4 * X5 + X0 * X3.pow_ref(2) * X4.pow_ref(2),
        5 + 4 * X0 + 6 * X4 * X5 + 3 * X6 * X5 + 4 * X0 * X4 + 3 * X0 * X6 + 6 * X0 * X3 + 6 * X0 * X2 + 6 * X6 * X4 * X5 + 2 * X0 * X4 * X5 + 4 * X0 * X6 * X5 + 3 * X0 * X2 * X5 + 3 * X0 * X6 * X4 + 5 * X0 * X3 * X4 + 6 * X0 * X2 * X4 + 4 * X0 * X6.pow_ref(2) + 3 * X0 * X3 * X6 + 3 * X0 * X2 * X6 + 2 * X0 * X6 * X4 * X5 + 6 * X0 * X3 * X4 * X5 + 5 * X0 * X1 * X4 * X5 + 6 * X0 * X6.pow_ref(2) * X5 + 2 * X0 * X3 * X6 * X5 + 2 * X0 * X2 * X6 * X5 + 6 * X0 * X1 * X6 * X5 + 6 * X0 * X2 * X3 * X5 + 6 * X0 * X3 * X4.pow_ref(2) + 4 * X0 * X6.pow_ref(2) * X4 + 6 * X0 * X3 * X6 * X4 + 3 * X0 * X2 * X6 * X4 + 4 * X0 * X3 * X6 * X4 * X5 + 5 * X0 * X1 * X6 * X4 * X5 + 6 * X0 * X3.pow_ref(2) * X4 * X5 + 5 * X0 * X2 * X3 * X4 * X5 + 3 * X0 * X3 * X6 * X4.pow_ref(2) + 6 * X0 * X3.pow_ref(2) * X4.pow_ref(2) * X5.clone(),
        2 + 2 * X0 + 4 * X0 * X4 + 2 * X0 * X6 + 5 * X0 * X4 * X5 + 2 * X0 * X6 * X5 + 4 * X0 * X2 * X5 + 2 * X0 * X4.pow_ref(2) + 4 * X0 * X6 * X4 + 4 * X0 * X6.pow_ref(2) + 2 * X6 * X4 * X5.pow_ref(2) + 4 * X0 * X6 * X4 * X5 + X0 * X3 * X4 * X5 + X0 * X2 * X4 * X5 + 3 * X0 * X6.pow_ref(2) * X5 + 2 * X0 * X3 * X6 * X5 + 4 * X0 * X2 * X6 * X5 + 2 * X0 * X6 * X4.pow_ref(2) + X0 * X6.pow_ref(2) * X4 + 3 * X0 * X6 * X4 * X5.pow_ref(2) + 2 * X0 * X6.pow_ref(2) * X5.pow_ref(2) + 3 * X0 * X2 * X6 * X5.pow_ref(2) + X0 * X3 * X4.pow_ref(2) * X5 + X0 * X6.pow_ref(2) * X4 * X5 + X0 * X3 * X6 * X4 * X5 + 6 * X0 * X2 * X6 * X4 * X5 + 4 * X0 * X6.pow_ref(2) * X4.pow_ref(2) + 6 * X0 * X3 * X6 * X4 * X5.pow_ref(2) + 4 * X0 * X1 * X6 * X4 * X5.pow_ref(2) + 4 * X0 * X2 * X3 * X4 * X5.pow_ref(2) + 6 * X0 * X3 * X6 * X4.pow_ref(2) * X5 + 2 * X0 * X3.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(2),
        4 + 5 * X0 * X4 * X5 + 6 * X0 * X6 * X5 + 5 * X0 * X4.pow_ref(2) * X5 + 3 * X0 * X6 * X4 * X5 + 3 * X0 * X6.pow_ref(2) * X5 + 6 * X0 * X6 * X4 * X5.pow_ref(2) + 5 * X0 * X2 * X4 * X5.pow_ref(2) + X0 * X6.pow_ref(2) * X5.pow_ref(2) + 6 * X0 * X2 * X6 * X5.pow_ref(2) + 4 * X0 * X6 * X4.pow_ref(2) * X5 + 2 * X0 * X6.pow_ref(2) * X4 * X5 + 5 * X0 * X3 * X4.pow_ref(2) * X5.pow_ref(2) + 3 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(2) + 5 * X0 * X3 * X6 * X4 * X5.pow_ref(2) + 4 * X0 * X2 * X6 * X4 * X5.pow_ref(2) + 6 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5 + 4 * X0 * X3 * X6 * X4.pow_ref(2) * X5.pow_ref(2),
        4 + 4 * X0 * X4.pow_ref(2) * X5.pow_ref(2) + X0 * X6 * X4 * X5.pow_ref(2) + X0 * X6.pow_ref(2) * X5.pow_ref(2) + 5 * X0 * X6 * X4.pow_ref(2) * X5.pow_ref(2) + 6 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(2) + 3 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(3) + 4 * X0 * X2 * X6 * X4 * X5.pow_ref(3) + 6 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(2) + 4 * X0 * X3 * X6 * X4.pow_ref(2) * X5.pow_ref(3),
        5 * X0 * X6 * X4.pow_ref(2) * X5.pow_ref(3) + 6 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(3) + 5 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(3),
        2 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(4)
    ]);

    let start = std::time::Instant::now();
    let gb = buchberger::<_, _, true>(&ring, basis, DegRevLex);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    assert_eq!(130, gb.len());
}