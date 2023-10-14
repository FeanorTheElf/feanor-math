use crate::ring::*;
use crate::rings::multivariate::*;
use crate::vector::*;

fn S<P, O>(ring: P, f1: &El<P>, f2: &El<P>, order: O) -> El<P> 
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        O: MonomialOrder + Copy
{
    let f1_lm = ring.lm(f1, order).unwrap();
    let f2_lm = ring.lm(f2, order).unwrap();
    let mut lcm: Monomial<_> = f1_lm.clone();
    lcm.lcm_assign(&f2_lm);
    let mut f1_factor = lcm.clone();
    f1_factor.div_assign(f1_lm);
    let mut f2_factor = lcm;
    f2_factor.div_assign(f2_lm);
    let mut f1_scaled = ring.clone_el(f1);
    ring.mul_monomial(&mut f1_scaled, &f1_factor);
    let mut f2_scaled = ring.clone_el(f2);
    ring.mul_monomial(&mut f2_scaled, &f2_factor);
    return ring.sub(f1_scaled, f2_scaled);
}

pub fn f4<P, O>(ring: P, mut basis: Vec<El<P>>, order: O) -> Vec<El<P>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        O: MonomialOrder + Copy
{
    let open = (0..basis.len()).flat_map(|i| (0..basis.len()).map(move |j| (i, j)))
        .filter(|(i, j)| i != j)
        .filter(|(i, j)| !ring.lm(basis.at(*i), order).unwrap().is_coprime(ring.lm(basis.at(*j), order).unwrap()))
        .collect::<Vec<_>>();
    while open.len() > 0 {
        
    }
    return basis;
}