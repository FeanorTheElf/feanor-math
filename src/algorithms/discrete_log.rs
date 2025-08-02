
use oorandom::Rand64;

use crate::algorithms::eea::inv_crt;
use crate::algorithms::int_bisect::root_floor;
use crate::algorithms::int_factor::factor;
use crate::algorithms::linsolve::LinSolveRingStore;
use crate::algorithms::linsolve::SolveResult;
use crate::algorithms::lll::exact::lll;
use crate::algorithms::matmul::ComputeInnerProduct;
use crate::algorithms::matmul::MatmulAlgorithm;
use crate::algorithms::matmul::STANDARD_MATMUL;
use crate::algorithms::sqr_mul::generic_abs_square_and_multiply;
use crate::assert_matrix_eq;
use crate::divisibility::DivisibilityRingStore;
use crate::field::FieldStore;
use crate::homomorphism::Homomorphism;
use crate::integer::int_cast;
use crate::integer::BigIntRing;
use crate::matrix::AsPointerToSlice;
use crate::matrix::OwnedMatrix;
use crate::matrix::Submatrix;
use crate::matrix::SubmatrixMut;
use crate::matrix::TransposableSubmatrix;
use crate::matrix::TransposableSubmatrixMut;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::finite::FiniteRingStore;
use crate::rings::rational::RationalField;
use crate::ordered::OrderedRingStore;

#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;

use std::alloc::Global;
use std::array::from_fn;
use std::hash::Hash;
use std::collections::HashMap;
use std::hash::Hasher;
use std::mem::replace;

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

pub trait DlogCapableGroup {
    type Element;

    fn clone_el(&self, x: &Self::Element) -> Self::Element;
    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool;
    fn op(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element;
    fn inv(&self, x: &Self::Element) -> Self::Element;
    fn identity(&self) -> Self::Element;
    fn hash<H: Hasher>(&self, x: &Self::Element, hasher: &mut H);

    fn pow(&self, x: &Self::Element, e: i64) -> Self::Element {
        let res = generic_abs_square_and_multiply(
            self.clone_el(x), 
            &e, 
            ZZ, 
            |a| self.op(self.clone_el(&a), &a), 
            |a, b| self.op(b, &a), 
            self.identity()
        );
        if e >= 0 { res } else { self.inv(&res) }
    }

    fn is_identity(&self, x: &Self::Element) -> bool {
        self.eq_el(x, &self.identity())
    }
}

pub struct GeneratingSet<G: DlogCapableGroup> {
    generators: Vec<G::Element>,
    order: El<BigIntRing>,
    order_factorization: Vec<(i64, usize)>,
    /// the `(i, j)`-th entry has rows that form a basis of the relation lattice of
    /// the set `n/pi^j g1, ..., n/pi^j gk` (where `n` is the order of the group, 
    /// and the `pi^ei` are its prime power factors)
    scaled_relation_lattices: Vec<Vec<OwnedMatrix<i64>>>,
    /// the `(i, j, k)`-th entry contains `sum_l row[l] n/pi^(j + 1) gl`, where
    /// `row` is the `k`-th row of `scaled_relation_lattice[i, j]`; These values
    /// are important, since they form a basis of the `p`-torsion subgroup of
    /// `< n/pi^(j + 1) g1, ..., n/pi^(j + 1) gk >`
    scaled_generating_sets: Vec<Vec<Vec<G::Element>>>
}

impl<G: DlogCapableGroup> GeneratingSet<G> {

    pub fn new(group: &G, order: El<BigIntRing>, generators: Vec<G::Element>) -> Self {
        let n = generators.len();
        if n == 0 {
            return Self {
                generators: Vec::new(),
                order: BigIntRing::RING.clone_el(&order),
                order_factorization: factor(BigIntRing::RING, order).into_iter().map(|(p, e)| (int_cast(p, ZZ, BigIntRing::RING), e)).collect(),
                scaled_generating_sets: Vec::new(),
                scaled_relation_lattices: Vec::new()
            };
        } else {
            let mut result = Self::new(group, order, Vec::new());
            for g in generators {
                result = result.add_generator(group, g);
            }
            return result;
        }
    }

    pub fn add_generator(&self, group: &G, new_gen_base: G::Element) -> Self {
        assert!(group.is_identity(&group.pow(&new_gen_base, int_cast(BigIntRing::RING.clone_el(&self.order), ZZ, BigIntRing::RING))));

        let mut scaled_relation_lattices = Vec::new();
        let mut scaled_generating_sets = Vec::new();
        for p_idx in 0..self.order_factorization.len() {
            
            let (p, e) = self.order_factorization[p_idx];
            let power = BigIntRing::RING.checked_div(
                &self.order, 
                &BigIntRing::RING.pow(int_cast(p, BigIntRing::RING, ZZ), e)
            ).unwrap();
            let power = int_cast(power, ZZ, BigIntRing::RING);
            let gens = self.generators.iter().map(|g| group.pow(g, power)).collect::<Vec<_>>();
            let new_gen = group.pow(&new_gen_base, power);
            
            let n = self.generators.len();

            let mut main_relation_matrix = OwnedMatrix::zero(n + 1, n + 1, ZZ);
            for i in 0..n {
                for j in 0..n {
                    *main_relation_matrix.at_mut(i, j) = *self.scaled_relation_lattices[p_idx][e].at(i, j);
                }
            }
            *main_relation_matrix.at_mut(n, n) = -ZZ.pow(p, e);
            for k in 0..e {
                if let Some(dlog) = self.dlog(group, &group.pow(&new_gen, ZZ.pow(p, k))) {
                    *main_relation_matrix.at_mut(n, n) = -ZZ.pow(p, k);
                    for j in 0..n {
                        *main_relation_matrix.at_mut(n, j) = dlog[j];
                    }
                    break;
                }
            }

            let mut result = Vec::with_capacity(e + 1);
            result.push(main_relation_matrix);
            for _ in 0..e {
                result.push(relation_lattice_basis_downscale_p(result.last().unwrap().data(), p));
            }
            result.reverse();
            scaled_relation_lattices.push(result);
            
            let mut generating_sets = Vec::new();
            for i in 0..e {
                let generating_set = scaled_relation_lattices.last().unwrap()[i].data().row_iter().map(|row|
                    (0..n).fold(group.pow(&new_gen, row[n] * ZZ.pow(p, e - i - 1)), |current, j| 
                        group.op(current, &group.pow(&gens[j], row[j] * ZZ.pow(p, e - i - 1)))
                    )
                ).collect::<Vec<_>>();
                generating_sets.push(generating_set);
            }
            scaled_generating_sets.push(generating_sets);
        }
        
        return Self {
            generators: self.generators.iter().map(|g| group.clone_el(g)).chain([new_gen_base].into_iter()).collect(),
            order: BigIntRing::RING.clone_el(&self.order),
            order_factorization: self.order_factorization.clone(),
            scaled_generating_sets: scaled_generating_sets,
            scaled_relation_lattices: scaled_relation_lattices
        };
    }

    /// 
    /// # Algorithm
    ///  
    /// We are working over `G = ord/p^e global_group`, in which every element
    /// has order dividing `p^e`. Clearly, it is generated by the global generators,
    /// scaled by `ord/p^e`.
    /// 
    /// We want to compute a dlog of `x` w.r.t. `g1, ..., gn`. For this, we use the exact sequence
    /// ```text
    ///   0  ->  H  ->  G  ->  G/H  ->  0
    /// ```
    /// where `H = { a in G | p a = 0 }` is the `p`-torsion subgroup. Note that the
    /// power-of-`p` map gives an isomorphism `G/H -> pG`, which allows us to recursively
    /// solve dlog in `G/H`. Hence, we want to solve dlog in `H`, which we can do using
    /// the baby-giant step method - if we can find a generating set of `H`. We find it
    /// using the already provided basis of the relation modulo of the generators.
    /// 
    fn padic_dlog(&self, group: &G, p_idx: usize, e: usize, target: &G::Element) -> Option<Vec<i64>> {
        
        let n = self.generators.len();
        assert!(n > 0);
        if e == 0 {
            debug_assert!(group.is_identity(target));
            return Some((0..n).map(|_| 0).collect());
        }

        let p = self.order_factorization[p_idx].0;
        debug_assert!(group.is_identity(&group.pow(target, ZZ.pow(p, e))));

        let power = BigIntRing::RING.checked_div(
            &self.order, 
            &BigIntRing::RING.pow(int_cast(p, BigIntRing::RING, ZZ), e)
        ).unwrap();
        let power = int_cast(power, ZZ, BigIntRing::RING);
        let gens = self.generators.iter().map(|g| group.pow(g, power)).collect::<Vec<_>>();
        
        unsafe { debugit::debugit!("gens", &gens) }
        unsafe { debugit::debugit!("target", &target) }
        unsafe { debugit::debugit!("(p, e)", &(p, e)) }

        // here we use the power-of-`p` map and the fact that `G/H ~ pG` to compute the dlog in `G/H`
        let G_mod_H_dlog = self.padic_dlog(
            group,
            p_idx,
            e - 1,
            &group.pow(target, p)
        )?;
        debug_assert!(group.eq_el(
            &group.pow(target, p),
            &(0..n).fold(group.identity(), |current, i| group.op(current, &group.pow(&gens[i], p * G_mod_H_dlog[i])))
        ));

        // delta is now in H, i.e. is a p-torsion element
        let delta = (0..n).fold(group.clone_el(target), |current, i|
            group.op(current, &group.pow(&gens[i], -G_mod_H_dlog[i]))
        );
        debug_assert!(group.is_identity(&group.pow(&delta, p)));

        let H_generators = &self.scaled_generating_sets[p_idx][e - 1];
        let H_dlog_wrt_H_gens = baby_giant_step(
            group, 
            delta, 
            &H_generators, 
            &(0..n).map(|_| int_cast(p, BigIntRing::RING, ZZ)).collect::<Vec<_>>()
        )?;
        let H_dlog = {
            let mut result = (0..n).map(|_| 0).collect::<Vec<_>>();
            STANDARD_MATMUL.matmul(
                TransposableSubmatrix::from(Submatrix::from_1d(&H_dlog_wrt_H_gens, 1, n)),
                TransposableSubmatrix::from(self.scaled_relation_lattices[p_idx][e - 1].data()),
                TransposableSubmatrixMut::from(SubmatrixMut::from_1d(&mut result, 1, n)),
                ZZ
            );
            result
        };

        let result = G_mod_H_dlog.into_iter().zip(H_dlog.into_iter()).map(|(x, y)| x + y).collect::<Vec<_>>();
        debug_assert!(group.eq_el(
            target,
            &(0..n).fold(group.identity(), |current, i| group.op(current, &group.pow(&gens[i], result[i])))
        ));

        return Some(result);
    }

    pub fn dlog(&self, group: &G, target: &G::Element) -> Option<Vec<i64>> {
        
        let n = self.generators.len();
        if n == 0 {
            return if group.is_identity(target) { Some(Vec::new()) } else { None };
        }

        let mut current_dlog = (0..n).map(|_| 0).collect::<Vec<_>>();
        let mut current_order = (0..n).map(|_| 1).collect::<Vec<_>>();

        for p_idx in 0..self.order_factorization.len() {
            let (p, e) = self.order_factorization[p_idx];
            let power = BigIntRing::RING.checked_div(
                &self.order, 
                &BigIntRing::RING.pow(int_cast(p, BigIntRing::RING, ZZ), e)
            ).unwrap();
            let power = int_cast(power, ZZ, BigIntRing::RING);
            let padic_dlog = self.padic_dlog(group, p_idx, e, &group.pow(target, power))?;
            for j in 0..n {
                current_dlog[j] = inv_crt(current_dlog[j], padic_dlog[j], &current_order[j], &ZZ.pow(p, e), ZZ);
                current_order[j] *= ZZ.pow(p, e);
            }
        }

        return Some(current_dlog);
    }
}

pub struct AddGroup<R: RingStore>(pub R);

pub struct ProdGroup<G: DlogCapableGroup, const N: usize>(G);

impl<R: RingStore> DlogCapableGroup for AddGroup<R>
    where R::Type: HashableElRing
{
    type Element = El<R>;

    fn clone_el(&self, x: &Self::Element) -> Self::Element { self.0.clone_el(x) }
    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool { self.0.eq_el(lhs, rhs) }
    fn op(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element { self.0.add_ref_snd(lhs, rhs)}
    fn inv(&self, x: &Self::Element) -> Self::Element { self.0.negate(self.0.clone_el(x)) }
    fn identity(&self) -> Self::Element { self.0.zero() }
    fn hash<H: Hasher>(&self, x: &Self::Element, hasher: &mut H) { self.0.hash(x, hasher) }
}

impl<G: DlogCapableGroup, const N: usize> DlogCapableGroup for ProdGroup<G, N> {
    type Element = [G::Element; N];

    fn clone_el(&self, x: &Self::Element) -> Self::Element { from_fn(|i| self.0.clone_el(&x[i])) }
    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool { (0..N).all(|i| self.0.eq_el(&lhs[i], &rhs[i])) }
    fn inv(&self, x: &Self::Element) -> Self::Element { from_fn(|i| self.0.inv(&x[i])) }
    fn identity(&self) -> Self::Element { from_fn(|_| self.0.identity()) }
    fn hash<H: Hasher>(&self, x: &Self::Element, hasher: &mut H) { x.into_iter().for_each(|x| self.0.hash(x, hasher)) }
    
    fn op(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element { 
        let mut it = lhs.into_iter().zip(rhs.into_iter()).map(|(l, r)| self.0.op(l, r));
        return from_fn(|_| it.next().unwrap());
    }
}

struct HashableGroupEl<'a, G: DlogCapableGroup> {
    group: &'a G,
    el: G::Element
}

impl<'a, G: DlogCapableGroup> HashableGroupEl<'a, G> {
    fn new(group: &'a G, el: G::Element) -> Self {
        Self { group, el }
    }
}

impl<'a, G: DlogCapableGroup> PartialEq for HashableGroupEl<'a, G> {
    fn eq(&self, other: &Self) -> bool {
        self.group.eq_el(&self.el, &other.el)
    }
}

impl<'a, G: DlogCapableGroup> Eq for HashableGroupEl<'a, G> {}

impl<'a, G: DlogCapableGroup> Hash for HashableGroupEl<'a, G> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.group.hash(&self.el, state)
    }
}

///
/// Computes the a vector `k` with entries `1 <= k[i] <= dlog_bounds[i]` such that
/// `generators^k = value` (`generators` is a list of elements of an abelian group).
/// 
/// If there is no such vector, then `None` is returned. If there are multiple such
/// vectors, any one of them is returned. In the 1d-case, it is guaranteed that this
/// is the smallest one, but in the multidimensional case, no such guarantee can be made
/// (in particular, the vector in general won't be the shortest one w.r.t. any natural
/// ordering like lex or degrevlex).
/// 
/// Note: The vector `k` is required to have positive entries. In particular, this
/// function won't return the zero vector if the given element is the identity.
/// This can have unexpected consequences, like
/// ```
/// # use feanor_math::algorithms::discrete_log;
/// assert_eq!(Some(vec![1]), baby_giant_step(0, &[0], &[1000], |a, b| a + b, 0));
/// ```
/// 
/// # Implementation notes
/// 
/// The complexity of the algorithm is `O(sqrt(prod_i dlog_bounds[i]))`. 
/// Thus, when possible, `order_bound[i]` should be the order of `generators[i]`
/// in the group.
/// 
/// Why do we need a group? Because we search for collisions `ab = ac`, and assume
/// that this implies `b = c`. So actually,  a cancelable abelian monoid would be sufficient...
/// 
/// Why don't we use Pollard's rhos? Because Pollard's rho cannot deterministically
/// detect the case that `value` is not in the subgroup generated by `generators`.
/// It can do so with high probability, but only if the used hash function satisfies
/// certain properties. With BSGS, the correctness does not depend on the used hash
/// function (although performance does, of course).
/// 
/// # Example
/// 
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::wrapper::*;
/// # use feanor_math::algorithms::discrete_log::*;
/// let ring = Zn::new(17);
/// let x = RingElementWrapper::new(&ring, ring.int_hom().map(9));
/// let one = RingElementWrapper::new(&ring, ring.one());
/// assert_eq!(Some((0, Some(8))), baby_giant_step(one.clone(), &x, 1000, |a, b| a * b, one));
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn baby_giant_step<G>(group: &G, value: G::Element, generators: &[G::Element], dlog_bounds: &[El<BigIntRing>]) -> Option<Vec<i64>> 
    where G: DlogCapableGroup
{
    let n = generators.len();
    assert_eq!(n, dlog_bounds.len());
    if generators.len() == 0 {
        if group.is_identity(&value) {
            return Some(Vec::new());
        } else {
            return None;
        }
    }
    let ns = dlog_bounds.iter().map(|n| int_cast(root_floor(BigIntRing::RING, n.clone(), 2), ZZ, BigIntRing::RING) + 1).collect::<Vec<_>>();
    let count = int_cast(BigIntRing::RING.prod(ns.iter().map(|n| int_cast(*n, BigIntRing::RING, ZZ))), ZZ, BigIntRing::RING);
    let mut baby_step_table: HashMap<HashableGroupEl<_>, i64> = HashMap::with_capacity(count as usize);
    
    // fill baby step table
    {
        let mut current_els = (0..n).map(|_| group.clone_el(&value)).collect::<Vec<_>>();
        let mut current_idxs = (0..n).map(|_| 0).collect::<Vec<_>>();
        for idx in 0..count {
            _ = baby_step_table.insert(HashableGroupEl::new(group, group.clone_el(&current_els[n - 1])), idx);

            let mut i = n - 1;
            while current_idxs[i] == ns[i] - 1 {
                if i == 0 {
                    assert!(idx + 1 == count);
                    break;
                }
                current_idxs[i] = 0;
                i -= 1;
            }
            current_idxs[i] += 1;
            current_els[i] = group.op(replace(&mut current_els[i], group.identity()), &generators[i]);
            for j in (i + 1)..n {
                current_els[j] = group.clone_el(&current_els[i]);
            }
        }
    }

    let giant_steps = generators.iter().zip(ns.iter()).map(|(g, n)| group.pow(g, *n)).collect::<Vec<_>>();
    // iterate through giant steps
    {
        let start_el = giant_steps.iter().fold(group.identity(), |x, y| group.op(x, y));
        let mut current_els = (0..n).map(|_| group.clone_el(&start_el)).collect::<Vec<_>>();
        let mut current_idxs = (0..n).map(|_| 1).collect::<Vec<_>>();
        for idx in 0..count {
            if let Some(bs_idx) = baby_step_table.get(&HashableGroupEl::new(group, group.clone_el(&current_els[n - 1]))) {
                let mut bs_idx = *bs_idx;
                let mut result = current_idxs.clone();
                for j in (0..n).rev() {
                    let bs_idxs_j = bs_idx % ns[j];
                    bs_idx = bs_idx / ns[j];
                    result[j] = result[j] * ns[j] - bs_idxs_j;
                }
                if (0..dlog_bounds.len()).all(|j| BigIntRing::RING.is_leq(&int_cast(result[j], BigIntRing::RING, ZZ), &dlog_bounds[j])) {
                    debug_assert_eq!(n, result.len());
                    return Some(result);
                }
            }

            let mut i = n - 1;
            while current_idxs[i] == ns[i] {
                if i == 0 {
                    assert!(idx + 1 == count);
                    break;
                }
                current_idxs[i] = 1;
                i -= 1;
            }
            current_idxs[i] += 1;
            current_els[i] = group.op(replace(&mut current_els[i], group.identity()), &giant_steps[i]);
            for j in (i + 1)..n {
                current_els[j] = group.clone_el(&current_els[i]);
            }
        }
    }

    return None;
}

///
/// Takes a matrix whose rows form a basis of the relation lattice w.r.t. group
/// elements `g1, ..., gn` and computes a matrix whose rows form a basis of the relation
/// lattice of `p g1, ..., p gn`.
/// 
fn relation_lattice_basis_downscale_p<V>(basis: Submatrix<V, i64>, p: i64) -> OwnedMatrix<i64>
    where V: AsPointerToSlice<i64>
{
    let n = basis.row_count();
    assert_eq!(n, basis.col_count());
    let QQ = RationalField::new(BigIntRing::RING);
    let ZZ_to_QQ = QQ.inclusion().compose(QQ.base_ring().can_hom(&ZZ).unwrap());
    let as_ZZ = |x| int_cast(BigIntRing::RING.checked_div(QQ.num(x), QQ.den(x)).unwrap(), ZZ, BigIntRing::RING);

    let mut dual_basis = OwnedMatrix::identity(n, 2 * n, &QQ);
    let mut Binv = dual_basis.data_mut().submatrix(0..n, n..(2 * n));
    let mut rhs = OwnedMatrix::identity(n, n, &QQ);
    QQ.solve_right(
        OwnedMatrix::from_fn(n, n, |i, j| ZZ_to_QQ.map_ref(basis.at(i, j))).data_mut(), 
        rhs.data_mut(),
        Binv.reborrow()
    ).assert_solved();
    Binv.reborrow().row_iter().flat_map(|row| row.iter_mut()).for_each(|x| ZZ_to_QQ.mul_assign_map(x, p));

    let mut identity = OwnedMatrix::identity(n, n, &QQ);
    lll(
        &QQ, 
        identity.data(), 
        dual_basis.data_mut(), 
        &QQ.div(&ZZ_to_QQ.map(9), &ZZ_to_QQ.map(10)),
        Global, 
        false
    );

    let mut result = rhs;
    QQ.solve_right(
        dual_basis.data_mut().submatrix(0..n, n..(2 * n)),
        identity.data_mut(),
        result.data_mut()
    ).assert_solved();

    return OwnedMatrix::from_fn(n, n, |i, j| as_ZZ(result.at(i, j)));
}

#[cfg(test)]
use crate::rings::zn::zn_static::Zn;

#[test]
fn test_baby_giant_step() {
    for base_bound in [21, 26, 31, 37] {
        let dlog_bound = [int_cast(base_bound, BigIntRing::RING, ZZ)];
        let G = AddGroup(ZZ);
        assert_eq!(
            Some(vec![6]), 
            baby_giant_step(&G, 6, &[1], &dlog_bound)
        );
        assert_eq!(
            None, 
            baby_giant_step(&G, 0, &[1], &dlog_bound)
        );

        let G = AddGroup(Zn::<20>::RING);
        assert_eq!(
            Some(vec![20]), 
            baby_giant_step(&G, 0, &[1], &dlog_bound)
        );
        assert_eq!(
            Some(vec![10]), 
            baby_giant_step(&G, 10, &[1], &dlog_bound)
        );
        assert_eq!(
            Some(vec![5]), 
            baby_giant_step(&G, 0, &[16], &dlog_bound)
        );
        
        let G = ProdGroup(AddGroup(Zn::<20>::RING));
        let dlog_bound: [_; 2] = from_fn(|_| int_cast(base_bound, BigIntRing::RING, ZZ));
        assert_eq!(
            Some(vec![10, 10]), 
            baby_giant_step(&G, [10, 10], &[[1, 0], [0, 1]], &dlog_bound)
        );
        assert_eq!(
            Some(vec![8, 14]), 
            baby_giant_step(&G, [8, 10], &[[1, 2], [0, 1]], &dlog_bound)
        );
    }
    
    let G = AddGroup(ZZ);

    // the collision point is at 96
    assert_eq!(
        Some(vec![9 - 1, 6 - 1]), 
        baby_giant_step(&G, 85, &[10, 1], &from_fn::<_, 2, _>(|_| int_cast(8, BigIntRing::RING, ZZ)))
    );
    // the collision point is at 105
    assert_eq!(
        Some(vec![10 - 2, 5 - 0]), 
        baby_giant_step(&G, 85, &[10, 1], &from_fn::<_, 2, _>(|_| int_cast(21, BigIntRing::RING, ZZ)))
    );
    // the collision point is at 90
    assert_eq!(
        Some(vec![6 - 0, 30 - 5]), 
        baby_giant_step(&G, 85, &[10, 1], &from_fn::<_, 2, _>(|_| int_cast(31, BigIntRing::RING, ZZ)))
    );
}


#[test]
fn test_padic_relation_lattice() {
    let G = AddGroup(Zn::<81>::RING);

    let gen_set = GeneratingSet::new(&G, int_cast(81, BigIntRing::RING, ZZ), vec![1]);
    assert_matrix_eq!(ZZ, [[-81]], gen_set.scaled_relation_lattices[0][4]);
    assert_matrix_eq!(ZZ, [[-27]], gen_set.scaled_relation_lattices[0][3]);
    assert_matrix_eq!(ZZ, [[-9]], gen_set.scaled_relation_lattices[0][2]);
    assert_matrix_eq!(ZZ, [[-3]], gen_set.scaled_relation_lattices[0][1]);
    assert_matrix_eq!(ZZ, [[1]], gen_set.scaled_relation_lattices[0][0]);

    let gen_set = GeneratingSet::new(&G, int_cast(81, BigIntRing::RING, ZZ), vec![3, 6]);
    let matrix = &gen_set.scaled_relation_lattices[0][4];
    assert_eq!(-27, *matrix.at(0, 0));
    assert_eq!(-1, *matrix.at(1, 1));
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([3, 6].iter())) % 81);

    let gen_set = GeneratingSet::new(&G, int_cast(81, BigIntRing::RING, ZZ), vec![3, 9]);
    let matrix = &gen_set.scaled_relation_lattices[0][4];
    assert_eq!(-27, *matrix.at(0, 0));
    assert_eq!(-1, *matrix.at(1, 1));
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([3, 9].iter())) % 81);

    let gen_set = GeneratingSet::new(&G, int_cast(81, BigIntRing::RING, ZZ), vec![6, 18, 9]);
    let matrix = &gen_set.scaled_relation_lattices[0][4];
    assert_eq!(-27, *matrix.at(0, 0));
    assert_eq!(-1, *matrix.at(1, 1));
    assert_eq!(-1, *matrix.at(2, 2));
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([6, 18, 9].iter())) % 81);
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(2).iter().zip([6, 18, 9].iter())) % 81);

    let G = ProdGroup(AddGroup(Zn::<81>::RING));

    let gen_set = GeneratingSet::new(&G, int_cast(81, BigIntRing::RING, ZZ), vec![[1, 4], [1, 1]]);
    let matrix = &gen_set.scaled_relation_lattices[0][4];
    assert_eq!(-81, *matrix.at(0, 0));
    assert_eq!(-27, *matrix.at(1, 1));
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([1, 1].iter())) % 81);
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([4, 1].iter())) % 81);
}

#[test]
fn random_test_dlog() {
    let ring = Zn::<1400>::RING;
    let i = ring.can_hom(&ZZ).unwrap();
    let mut rng = Rand64::new(0);
    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let G = ProdGroup(AddGroup(ring));
        let g1 = [ring.random_element(|| rng.rand_u64()), ring.random_element(|| rng.rand_u64()), ring.random_element(|| rng.rand_u64())];
        let g2 = [ring.random_element(|| rng.rand_u64()), ring.random_element(|| rng.rand_u64()), ring.random_element(|| rng.rand_u64())];
        let g3 = [ring.random_element(|| rng.rand_u64()), ring.random_element(|| rng.rand_u64()), ring.random_element(|| rng.rand_u64())];
        let gen_set = GeneratingSet::new(&G, int_cast(1400, BigIntRing::RING, ZZ), vec![g1, g2, g3]);

        let val = [ring.random_element(|| rng.rand_u64()), ring.random_element(|| rng.rand_u64()), ring.random_element(|| rng.rand_u64())];
        let dlog = gen_set.dlog(&G, &val);
        println!("{:?} * x + {:?} * y + {:?} * z = {:?} mod 1400", g1, g2, g3, val);
        if let Some(dlog) = dlog {
            for k in 0..3 {
                assert_el_eq!(ring, val[k], ring.sum([i.mul_map(g1[k], dlog[0]), i.mul_map(g2[k], dlog[1]), i.mul_map(g3[k], dlog[2])]));
            }
            println!("checked solution");
        } else {
            let mut gen_matrix = OwnedMatrix::from_fn(3, 3, |i, j| [g1, g2, g3][j][i]);
            let mut value = OwnedMatrix::from_fn(3, 1, |i, _| val[i]);
            let mut res = OwnedMatrix::zero(3, 1, ring);
            let solved = ring.solve_right(gen_matrix.data_mut(), value.data_mut(), res.data_mut());
            println!("[{}, {}, {}]", res.at(0, 0), res.at(1, 0), res.at(2, 0));
            if solved.is_solved() {
                for k in 0..3 {
                    assert_el_eq!(ring, val[k], ring.sum([ring.mul(g1[k], *res.at(0, 0)), ring.mul(g2[k], *res.at(1, 0)), ring.mul(g3[k], *res.at(2, 0))]));
                }
                assert!(solved == SolveResult::NoSolution);
            }
            println!("has no solution");
        }
    }
}