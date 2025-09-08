
use crate::algorithms::eea::inv_crt;
use crate::algorithms::eea::signed_gcd;
use crate::algorithms::int_bisect::root_floor;
use crate::algorithms::int_factor::factor;
use crate::algorithms::linsolve::smith::determinant_using_pre_smith;
use crate::algorithms::linsolve::smith::pre_smith;
use crate::algorithms::linsolve::LinSolveRingStore;
use crate::algorithms::lll::exact::lll;
use crate::algorithms::matmul::MatmulAlgorithm;
use crate::algorithms::matmul::STANDARD_MATMUL;
use crate::algorithms::sqr_mul::generic_abs_square_and_multiply;
use crate::divisibility::DivisibilityRing;
use crate::divisibility::DivisibilityRingStore;
use crate::iters::multi_cartesian_product;
use crate::matrix::transform::TransformTarget;
use crate::rings::finite::FiniteRingStore;
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
use crate::rings::rational::RationalField;
use crate::ordered::OrderedRingStore;
use crate::rings::zn::ZnRingStore;
use crate::rings::zn::zn_big;
use crate::rings::zn::ZnRing;
use crate::seq::VectorView;
#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;

use std::alloc::Global;
use std::array::from_fn;
use std::fmt::Debug;
use std::hash::Hash;
use std::collections::HashMap;
use std::hash::Hasher;
use std::mem::replace;

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;
const ZZbig: BigIntRing = BigIntRing::RING;

///
/// Trait for implementations of generic groups, for which only the group operation,
/// equality testing and computing hash values is supported.
/// 
/// These groups from the model for which most dlog algorithms have been developed.
/// Note that if your group is actually the additive group of a ring, it is very
/// likely that you can solve dlog much more efficiently by using [`crate::algorithms::linsolve`].
/// 
/// The design of this trait is a little bit like [`crate::ring::RingBase`], but
/// since it is only used locally, it is much less sophisticated.
/// 
#[stability::unstable(feature = "enable")]
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

///
/// Represents a subgroup of a [`DlogCapableGroup`] by a set of generators.
/// Supports computing discrete logarithms, i.e. representing a given element
/// as a combination of the generators.
/// 
/// Note that the used algorithms have a worst case complexity of `O(sqrt(ord^n))`
/// where `ord` is the given multiple of the orders of each generator, and `n`
/// is the number of generators. However, if `ord` is smooth, much faster algorithms
/// are used.
/// 
#[stability::unstable(feature = "enable")]
pub struct GeneratingSet<G: DlogCapableGroup> {
    generators: Vec<G::Element>,
    order_multiple: El<BigIntRing>,
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

    ///
    /// Creates a new [`GeneratingSet`] representing the subgroup generated
    /// by the given generators.
    /// 
    /// The value `order_multiple` should be a multiple of the order of every
    /// generator, including generators that will be added later on via
    /// [`GeneratingSet::add_generator()`].
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new(group: &G, order_multiple: El<BigIntRing>, generators: Vec<G::Element>) -> Self {
        let n = generators.len();
        if n == 0 {
            return Self {
                generators: Vec::new(),
                order_multiple: ZZbig.clone_el(&order_multiple),
                order_factorization: factor(ZZbig, order_multiple).into_iter().map(|(p, e)| (int_cast(p, ZZ, ZZbig), e)).collect(),
                scaled_generating_sets: Vec::new(),
                scaled_relation_lattices: Vec::new()
            };
        } else {
            let mut result = Self::new(group, order_multiple, Vec::new());
            for g in generators {
                result = result.add_generator(group, g);
            }
            return result;
        }
    }

    ///
    /// The number of elements in the subgroup generated by this generating set.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn subgroup_order(&self) -> El<BigIntRing> {
        let mut result = ZZbig.one();
        let n = self.generators.len();
        if n == 0 {
            return result;
        }
        for i in 0..self.order_factorization.len() {
            let (p, e) = self.order_factorization[i];
            let relation_lattice = self.scaled_relation_lattices[i][e].data();
            let Zpne = zn_big::Zn::new(ZZbig, ZZbig.pow(int_cast(p, ZZbig, StaticRing::<i64>::RING), e * n));
            let mod_pne = Zpne.can_hom(&StaticRing::<i64>::RING).unwrap();
            let relation_lattice_det = determinant_using_pre_smith(
                &Zpne, 
                OwnedMatrix::from_fn(relation_lattice.row_count(), relation_lattice.col_count(), |k, l| mod_pne.map(*relation_lattice.at(k, l))).data_mut(), 
                Global
            );
            ZZbig.mul_assign(&mut result, signed_gcd(ZZbig.clone_el(Zpne.modulus()), Zpne.smallest_positive_lift(relation_lattice_det), ZZbig));
        }
        return result;
    }

    ///
    /// Returns a multiple of the order of each element in the subgroup
    /// generated by this generating set.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn order_multiple(&self) -> &El<BigIntRing> {
        &self.order_multiple
    }

    ///
    /// Extends the generating set by an additional generator, which is likely
    /// to grow the represented subgroup.
    /// 
    /// The new generator must be of order dividing [`GeneratingSet::order_multiple()`].
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn add_generator(&self, group: &G, new_gen_base: G::Element) -> Self {
        assert!(group.is_identity(&group.pow(&new_gen_base, int_cast(ZZbig.clone_el(&self.order_multiple), ZZ, ZZbig))));

        let mut scaled_relation_lattices = Vec::new();
        let mut scaled_generating_sets = Vec::new();
        for p_idx in 0..self.order_factorization.len() {
            
            let (p, e) = self.order_factorization[p_idx];
            let power = ZZbig.checked_div(
                &self.order_multiple, 
                &ZZbig.pow(int_cast(p, ZZbig, ZZ), e)
            ).unwrap();
            let power = int_cast(power, ZZ, ZZbig);
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
                if let Some(dlog) = self.padic_dlog(group, p_idx, e, &group.pow(&new_gen, ZZ.pow(p, k))) {
                    *main_relation_matrix.at_mut(n, n) = -ZZ.pow(p, k);
                    for j in 0..n {
                        *main_relation_matrix.at_mut(n, j) = dlog[j];
                    }
                    break;
                }
            }
            debug_assert!(main_relation_matrix.data().row_iter().all(|row| group.is_identity(
                &(0..n).fold(group.pow(&new_gen, row[n]), |current, i| group.op(current, &group.pow(&gens[i], row[i])))
            )));

            let mut result = Vec::with_capacity(e + 1);
            result.push(main_relation_matrix);
            for _ in 0..e {
                result.push(Self::relation_lattice_basis_downscale_p(result.last().unwrap().data(), p));
            }
            result.reverse();
            scaled_relation_lattices.push(result);
            
            let mut generating_sets = Vec::new();
            for i in 0..e {
                let generating_set = scaled_relation_lattices.last().unwrap()[i].data().row_iter().map(|row| {
                    let result = (0..n).fold(group.pow(&new_gen, row[n] * ZZ.pow(p, e - i - 1)), |current, j| 
                        group.op(current, &group.pow(&gens[j], row[j] * ZZ.pow(p, e - i - 1)))
                    );
                    debug_assert!(group.is_identity(&group.pow(&result, p)));
                    result
                }).collect::<Vec<_>>();
                generating_sets.push(generating_set);
            }
            scaled_generating_sets.push(generating_sets);
        }
        
        return Self {
            generators: self.generators.iter().map(|g| group.clone_el(g)).chain([new_gen_base].into_iter()).collect(),
            order_multiple: ZZbig.clone_el(&self.order_multiple),
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
        if n == 0 {
            return if group.is_identity(target) { Some(Vec::new()) } else { None };
        } else if e == 0 {
            debug_assert!(group.is_identity(target));
            return Some((0..n).map(|_| 0).collect());
        }

        let p = self.order_factorization[p_idx].0;
        debug_assert!(group.is_identity(&group.pow(target, ZZ.pow(p, e))));

        let power = ZZbig.checked_div(
            &self.order_multiple, 
            &ZZbig.pow(int_cast(p, ZZbig, ZZ), e)
        ).unwrap();
        let power = int_cast(power, ZZ, ZZbig);
        let gens = self.generators.iter().map(|g| group.pow(g, power)).collect::<Vec<_>>();

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
            &(0..n).map(|_| int_cast(p, ZZbig, ZZ)).collect::<Vec<_>>()
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

    ///
    /// Computes a discrete logarithm of `target` w.r.t. the stored set
    /// if generators, or `None` if `target` is not in the subgroup generated by
    /// these generators
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn dlog(&self, group: &G, target: &G::Element) -> Option<Vec<i64>> {
        
        let n = self.generators.len();
        if n == 0 {
            return if group.is_identity(target) { Some(Vec::new()) } else { None };
        }

        let mut current_dlog = (0..n).map(|_| 0).collect::<Vec<_>>();
        let mut current_order = (0..n).map(|_| 1).collect::<Vec<_>>();

        for p_idx in 0..self.order_factorization.len() {
            let (p, e) = self.order_factorization[p_idx];
            let power = ZZbig.checked_div(
                &self.order_multiple, 
                &ZZbig.pow(int_cast(p, ZZbig, ZZ), e)
            ).unwrap();
            let power = int_cast(power, ZZ, ZZbig);
            let padic_dlog = self.padic_dlog(group, p_idx, e, &group.pow(target, power))?;
            for j in 0..n {
                current_dlog[j] = inv_crt(current_dlog[j], padic_dlog[j], &current_order[j], &ZZ.pow(p, e), ZZ);
                current_order[j] *= ZZ.pow(p, e);
            }
        }
        debug_assert!(group.eq_el(
            target,
            &(0..n).fold(group.identity(), |current, i| group.op(current, &group.pow(&self.generators[i], current_dlog[i])))
        ));

        return Some(current_dlog);
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

        let QQ = RationalField::new(ZZbig);
        let ZZ_to_QQ = QQ.inclusion().compose(QQ.base_ring().can_hom(&ZZ).unwrap());
        let as_ZZ = |x| int_cast(ZZbig.checked_div(QQ.num(x), QQ.den(x)).unwrap(), ZZ, ZZbig);

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

        let mut result_QQ = rhs;
        QQ.solve_right(
            dual_basis.data_mut().submatrix(0..n, n..(2 * n)),
            identity.data_mut(),
            result_QQ.data_mut()
        ).assert_solved();

        let result = OwnedMatrix::from_fn(n, n, |i, j| as_ZZ(result_QQ.at(i, j)));

        return result;
    }
}

impl<G: DlogCapableGroup> Clone for GeneratingSet<G>
    where G::Element: Clone
{
    fn clone(&self) -> Self {
        Self {
            generators: self.generators.clone(),
            order_factorization: self.order_factorization.clone(),
            order_multiple: self.order_multiple.clone(),
            scaled_generating_sets: self.scaled_generating_sets.clone(),
            scaled_relation_lattices: self.scaled_relation_lattices.iter().map(|x| x.iter().map(|x| x.clone_matrix(StaticRing::<i64>::RING)).collect()).collect()
        }
    }
}

impl<G: DlogCapableGroup> VectorView<G::Element> for GeneratingSet<G> {

    fn len(&self) -> usize {
        self.generators.len()
    }

    fn at(&self, i: usize) -> &G::Element {
        self.generators.at(i)
    }
}

impl<R> GeneratingSet<MultGroup<R>>
    where R: RingStore,
        R::Type: ZnRing + HashableElRing + DivisibilityRing
{
    pub fn for_zn(group: &MultGroup<R>, generators: Vec<<MultGroup<R> as DlogCapableGroup>::Element>) -> Self {
        let n = generators.len();
        if n == 0 {
            let n_factorization = factor(ZZbig, group.0.size(ZZbig).unwrap());
            let mut order_factorization = n_factorization.into_iter().flat_map(|(p, e)| 
                factor(ZZbig, ZZbig.sub_ref_fst(&p, ZZbig.one())).into_iter().chain([(p, e - 1)].into_iter())
            ).collect::<Vec<_>>();
            order_factorization.sort_unstable_by(|(pl, _), (pr, _)| ZZbig.cmp(pl, pr));
            order_factorization.dedup_by(|(p1, e1), (p2, e2)| if ZZbig.eq_el(p1, p2) {
                *e2 += *e1;
                true
            } else {
                false
            });
            order_factorization.retain(|(_, e)| *e > 0);
            let order = ZZbig.prod(order_factorization.iter().map(|(p, e)| ZZbig.pow(ZZbig.clone_el(p), *e)));
            let order_factorization = order_factorization.into_iter().map(|(p, e)| (int_cast(p, ZZ, ZZbig), e)).collect();
            return Self {
                generators: Vec::new(),
                order_multiple: order,
                order_factorization: order_factorization,
                scaled_generating_sets: Vec::new(),
                scaled_relation_lattices: Vec::new()
            };
        } else {
            let mut result = Self::for_zn(group, Vec::new());
            for g in generators {
                result = result.add_generator(group, g);
            }
            return result;
        }
    }
}

///
/// The additive group of a ring, implements [`DlogCapableGroup`].
/// 
/// Note that in most cases, it does not make much sense to compute
/// dlogs in the additive group of a ring using generic methods, since
/// algorithms as in [`crate::algorithms::linsolve`] will be much faster.
/// 
#[stability::unstable(feature = "enable")]
pub struct AddGroup<R: RingStore>(pub R);

///
/// The direct product of `N` copies of an underlying group, implements [`DlogCapableGroup`].
/// 
#[stability::unstable(feature = "enable")]
pub struct ProdGroup<G: DlogCapableGroup, const N: usize>(pub G);

///
/// The multiplicative group of a ring, implements [`DlogCapableGroup`].
/// 
#[stability::unstable(feature = "enable")]
pub struct MultGroup<R: RingStore>(pub R);

///
/// Elements from the multiplicative group of `R`.
/// 
#[stability::unstable(feature = "enable")]
pub struct MultGroupEl<R: RingStore>(El<R>);

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

impl<R: RingStore> Clone for MultGroup<R> 
    where R: Clone, R::Type: HashableElRing + DivisibilityRing
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<R: RingStore> Copy for MultGroup<R> 
    where R: Copy, R::Type: HashableElRing + DivisibilityRing
{}

impl<R: RingStore> Debug for MultGroup<R> 
    where R::Type: Debug + HashableElRing + DivisibilityRing
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?})*", self.0.get_ring())
    }
}

impl<R: RingStore> Clone for MultGroupEl<R> 
    where R::Type: HashableElRing + DivisibilityRing,
        El<R>: Clone
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<R: RingStore> Debug for MultGroupEl<R> 
    where R::Type: HashableElRing + DivisibilityRing,
        El<R>: Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl<R: RingStore> MultGroup<R> 
    where R::Type: HashableElRing + DivisibilityRing
{
    ///
    /// If `x` is contained in `R*`, returns a [`MultGroupEl`] representing
    /// `x`. Otherwise, `None` is returned.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn from_ring_el(&self, x: El<R>) -> Option<MultGroupEl<R>> {
        if self.0.is_unit(&x) {
            Some(MultGroupEl(x))
        } else {
            None
        }
    }

    ///
    /// Returns the ring element represented by the given group element.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn as_ring_el<'a>(&self, x: &'a MultGroupEl<R>) -> &'a El<R> {
        &x.0
    }
}

impl<R: RingStore> DlogCapableGroup for MultGroup<R> 
    where R::Type: HashableElRing + DivisibilityRing
{
    type Element = MultGroupEl<R>;

    fn clone_el(&self, x: &Self::Element) -> Self::Element { MultGroupEl(self.0.clone_el(&x.0)) }
    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool { self.0.eq_el(&lhs.0, &rhs.0) }
    fn inv(&self, x: &Self::Element) -> Self::Element { MultGroupEl(self.0.invert(&x.0).unwrap()) }
    fn identity(&self) -> Self::Element { MultGroupEl(self.0.one()) }
    fn hash<H: Hasher>(&self, x: &Self::Element, hasher: &mut H) { self.0.hash(&x.0, hasher) }
    fn op(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element { MultGroupEl(self.0.mul_ref_snd(lhs.0, &rhs.0)) }
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
/// # use feanor_math::algorithms::discrete_log::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::primitive_int::*;
/// let group = AddGroup(StaticRing::<i64>::RING);
/// assert_eq!(Some(vec![1]), baby_giant_step(&group, 0, &[0], &[BigIntRing::RING.power_of_two(10)]));
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
/// # use feanor_math::integer::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::wrapper::*;
/// # use feanor_math::algorithms::discrete_log::*;
/// let ring = Zn::new(17);
/// let group = MultGroup(ring);
/// let x = group.from_ring_el(ring.int_hom().map(9)).unwrap();
/// assert_eq!(Some(vec![3]), baby_giant_step(&group, group.pow(&x, 3), &[x], &[BigIntRing::RING.power_of_two(4)]));
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
    let ns = dlog_bounds.iter().map(|n| int_cast(root_floor(ZZbig, n.clone(), 2), ZZ, ZZbig) + 1).collect::<Vec<_>>();
    let count = int_cast(ZZbig.prod(ns.iter().map(|n| int_cast(*n, ZZbig, ZZ))), ZZ, ZZbig);
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
                if (0..dlog_bounds.len()).all(|j| ZZbig.is_leq(&int_cast(result[j], ZZbig, ZZ), &dlog_bounds[j])) {
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
/// Computes the discrete log in the group `(Z/nZ)*`.
/// 
/// This only called `finite_field_discrete_log` for backwards compatibility,
/// it now completely supports `(Z/nZ)*` for any `n`.
/// 
pub fn finite_field_discrete_log<R: RingStore>(value: El<R>, base: El<R>, Zn: R) -> Option<i64>
    where R::Type: ZnRing + HashableElRing
{
    let group = MultGroup(Zn);
    let gen_set = GeneratingSet::for_zn(&group, vec![group.from_ring_el(base).unwrap()]);
    return gen_set.dlog(&group, &group.from_ring_el(value).unwrap())
        .map(|res| res[0]);
}

///
/// Computes the multiplicative order in the group `(Z/nZ)*`.
/// 
pub fn multiplicative_order<R: RingStore>(x: El<R>, Zn: R) -> i64
    where R::Type: ZnRing + HashableElRing
{
    let group = MultGroup(Zn);
    let Zn = &group.0;
    let gen_set = GeneratingSet::for_zn(&group, Vec::new());

    let mut result = ZZbig.one();
    for (p, e) in &gen_set.order_factorization {
        let mut current = Zn.pow_gen(
            Zn.clone_el(&x), 
            &ZZbig.checked_div(&gen_set.order_multiple, &ZZbig.pow(int_cast(*p, ZZbig, ZZ), *e)).unwrap(), 
            ZZbig
        );
        while !Zn.is_one(&current) {
            current = Zn.pow(current, *p as usize);
            ZZbig.mul_assign(&mut result, int_cast(*p, ZZbig, ZZ));
        }
    }
    return int_cast(result, ZZ, ZZbig);
}

#[cfg(test)]
use crate::rings::zn::zn_static::Zn;
#[cfg(test)]
use oorandom::Rand64;
#[cfg(test)]
use crate::algorithms::linsolve::SolveResult;
#[cfg(test)]
use crate::algorithms::matmul::ComputeInnerProduct;
#[cfg(test)]
use crate::assert_matrix_eq;

#[test]
fn test_baby_giant_step() {
    for base_bound in [21, 26, 31, 37] {
        let dlog_bound = [int_cast(base_bound, ZZbig, ZZ)];
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
        let dlog_bound: [_; 2] = from_fn(|_| int_cast(base_bound, ZZbig, ZZ));
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
        baby_giant_step(&G, 85, &[10, 1], &from_fn::<_, 2, _>(|_| int_cast(8, ZZbig, ZZ)))
    );
    // the collision point is at 105
    assert_eq!(
        Some(vec![10 - 2, 5 - 0]), 
        baby_giant_step(&G, 85, &[10, 1], &from_fn::<_, 2, _>(|_| int_cast(21, ZZbig, ZZ)))
    );
    // the collision point is at 90
    assert_eq!(
        Some(vec![6 - 0, 30 - 5]), 
        baby_giant_step(&G, 85, &[10, 1], &from_fn::<_, 2, _>(|_| int_cast(31, ZZbig, ZZ)))
    );
}


#[test]
fn test_padic_relation_lattice() {
    let G = AddGroup(Zn::<81>::RING);

    let gen_set = GeneratingSet::new(&G, int_cast(81, ZZbig, ZZ), vec![1]);
    assert_matrix_eq!(ZZ, [[-81]], gen_set.scaled_relation_lattices[0][4]);
    assert_matrix_eq!(ZZ, [[-27]], gen_set.scaled_relation_lattices[0][3]);
    assert_matrix_eq!(ZZ, [[-9]], gen_set.scaled_relation_lattices[0][2]);
    assert_matrix_eq!(ZZ, [[-3]], gen_set.scaled_relation_lattices[0][1]);
    assert_matrix_eq!(ZZ, [[1]], gen_set.scaled_relation_lattices[0][0]);

    let gen_set = GeneratingSet::new(&G, int_cast(81, ZZbig, ZZ), vec![3, 6]);
    let matrix = &gen_set.scaled_relation_lattices[0][4];
    assert_eq!(-27, *matrix.at(0, 0));
    assert_eq!(-1, *matrix.at(1, 1));
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([3, 6].iter())) % 81);

    let gen_set = GeneratingSet::new(&G, int_cast(81, ZZbig, ZZ), vec![3, 9]);
    let matrix = &gen_set.scaled_relation_lattices[0][4];
    assert_eq!(-27, *matrix.at(0, 0));
    assert_eq!(-1, *matrix.at(1, 1));
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([3, 9].iter())) % 81);

    let gen_set = GeneratingSet::new(&G, int_cast(81, ZZbig, ZZ), vec![6, 18, 9]);
    let matrix = &gen_set.scaled_relation_lattices[0][4];
    assert_eq!(-27, *matrix.at(0, 0));
    assert_eq!(-1, *matrix.at(1, 1));
    assert_eq!(-1, *matrix.at(2, 2));
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([6, 18, 9].iter())) % 81);
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(2).iter().zip([6, 18, 9].iter())) % 81);

    let G = ProdGroup(AddGroup(Zn::<81>::RING));

    let gen_set = GeneratingSet::new(&G, int_cast(81, ZZbig, ZZ), vec![[1, 4], [1, 1]]);
    let matrix = &gen_set.scaled_relation_lattices[0][4];
    assert_eq!(-81, *matrix.at(0, 0));
    assert_eq!(-27, *matrix.at(1, 1));
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([1, 1].iter())) % 81);
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([4, 1].iter())) % 81);

    let G = ProdGroup(AddGroup(Zn::<8>::RING));

    let gen_set = GeneratingSet::new(&G, int_cast(8, ZZbig, ZZ), vec![[6, 3, 5], [6, 2, 6], [4, 5, 7]]);
    let matrix = &gen_set.scaled_relation_lattices[0][3];
    assert_eq!(-8, *matrix.at(0, 0));
    assert_eq!(-4, *matrix.at(1, 1));
    assert_eq!(-2, *matrix.at(2, 2));
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([6, 6, 4].iter())) % 8);
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([3, 2, 5].iter())) % 8);
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([5, 6, 7].iter())) % 8);
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(2).iter().zip([6, 6, 4].iter())) % 8);
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(2).iter().zip([3, 2, 5].iter())) % 8);
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(2).iter().zip([5, 6, 7].iter())) % 8);
}

#[test]
fn random_test_dlog() {
    let ring = Zn::<1400>::RING;
    let i = ring.can_hom(&ZZ).unwrap();
    let mut rng = Rand64::new(0);
    let G = ProdGroup(AddGroup(ring));
    let rand_gs = |rng: &mut Rand64| from_fn::<_, 3, _>(|_| ring.random_element(|| rng.rand_u64()));

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let gs = from_fn::<_, 3, _>(|_| rand_gs(&mut rng));
        let gen_set = GeneratingSet::new(&G, int_cast(1400, ZZbig, ZZ), gs.into());

        let coeffs = rand_gs(&mut rng);
        let val = (0..3).fold(G.identity(), |current, i| G.op(current, &G.pow(&gs[i], coeffs[i] as i64)));
        let dlog = gen_set.dlog(&G, &val);
        println!("{:?} * x + {:?} * y + {:?} * z = {:?} mod 1400", gs[0], gs[1], gs[2], val);
        if let Some(dlog) = dlog {
            for k in 0..3 {
                assert_el_eq!(ring, val[k], ring.sum([i.mul_map(gs[0][k], dlog[0]), i.mul_map(gs[1][k], dlog[1]), i.mul_map(gs[2][k], dlog[2])]));
            }
            println!("checked solution");
        }
    }

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let gs = from_fn::<_, 3, _>(|_| rand_gs(&mut rng));
        let gen_set = GeneratingSet::new(&G, int_cast(1400, ZZbig, ZZ), gs.into());

        let val = rand_gs(&mut rng);
        let dlog = gen_set.dlog(&G, &val);
        println!("{:?} * x + {:?} * y + {:?} * z = {:?} mod 1400", gs[0], gs[1], gs[2], val);
        if let Some(dlog) = dlog {
            for k in 0..3 {
                assert_el_eq!(ring, val[k], ring.sum([i.mul_map(gs[0][k], dlog[0]), i.mul_map(gs[1][k], dlog[1]), i.mul_map(gs[2][k], dlog[2])]));
            }
            println!("checked solution");
        } else {
            let mut gen_matrix = OwnedMatrix::from_fn(3, 3, |i, j| gs[j][i]);
            let mut value = OwnedMatrix::from_fn(3, 1, |i, _| val[i]);
            let mut res = OwnedMatrix::zero(3, 1, ring);
            let solved = ring.solve_right(gen_matrix.data_mut(), value.data_mut(), res.data_mut());
            println!("[{}, {}, {}]", res.at(0, 0), res.at(1, 0), res.at(2, 0));
            if solved.is_solved() {
                for k in 0..3 {
                    assert_el_eq!(ring, val[k], ring.sum([ring.mul(gs[0][k], *res.at(0, 0)), ring.mul(gs[1][k], *res.at(1, 0)), ring.mul(gs[2][k], *res.at(2, 0))]));
                }
                assert!(solved == SolveResult::NoSolution);
            }
            println!("has no solution");
        }
    }
}

#[test]
fn test_zn_dlog() {
    let ring = Zn::<51>::RING;
    let g1 = ring.int_hom().map(37);
    let g2 = ring.int_hom().map(35);

    assert_eq!(0, finite_field_discrete_log(ring.one(), g1, ring).unwrap() % 16);
    assert_eq!(0, finite_field_discrete_log(ring.one(), g2, ring).unwrap() % 2);
    assert_eq!(1, finite_field_discrete_log(g2, g2, ring).unwrap() % 2);
    for i in 0..16 {
        assert_eq!(i, finite_field_discrete_log(ring.pow(g1, i as usize), g1, ring).unwrap() % 16);
    }
    for i in 0..16 {
        assert_eq!(None, finite_field_discrete_log(ring.mul(ring.pow(g1, i as usize), g2), g1, ring));
    }

    assert_eq!(16, multiplicative_order(g1, ring));
    assert_eq!(8, multiplicative_order(ring.pow(g1, 2), ring));
    assert_eq!(4, multiplicative_order(ring.pow(g1, 4), ring));
    assert_eq!(2, multiplicative_order(ring.pow(g1, 8), ring));
    assert_eq!(2, multiplicative_order(g2, ring));
}

#[test]
fn test_zn_subgroup_size() {
    let ring = Zn::<153>::RING;
    let group = MultGroup(ring);
    let g1 = group.from_ring_el(ring.int_hom().map(2)).unwrap();
    let g2 = group.from_ring_el(ring.int_hom().map(37)).unwrap();

    let mut generating_set = GeneratingSet::for_zn(&group, Vec::new());
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(1), generating_set.subgroup_order());

    generating_set = generating_set.add_generator(&MultGroup(ring), group.clone_el(&g1));
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(24), generating_set.subgroup_order());
    
    generating_set = generating_set.add_generator(&MultGroup(ring), group.pow(&g1, 2));
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(24), generating_set.subgroup_order());
    
    generating_set = generating_set.add_generator(&MultGroup(ring), group.clone_el(&g2));
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(96), generating_set.subgroup_order());
    
    let generating_set = GeneratingSet::for_zn(&group, vec![g2]);
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(16), generating_set.subgroup_order());

}