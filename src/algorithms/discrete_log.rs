use std::fmt::Debug;

use crate::algorithms::eea::{inv_crt, signed_gcd};
use crate::algorithms::int_bisect::root_floor;
use crate::algorithms::int_factor::factor;
use crate::algorithms::linsolve::smith::{determinant_using_pre_smith, pre_smith};
use crate::algorithms::linsolve::LinSolveRingStore;
use crate::algorithms::lll::exact::lll;
use crate::algorithms::matmul::MatmulAlgorithm;
use crate::algorithms::matmul::STANDARD_MATMUL;
use crate::divisibility::DivisibilityRing;
use crate::divisibility::DivisibilityRingStore;
use crate::group::*;
use crate::group::HashableGroupEl;
use crate::group::MultGroup;
use crate::iters::multi_cartesian_product;
use crate::matrix::transform::TransformTarget;
use crate::pid::PrincipalIdealRingStore;
use crate::rings::finite::FiniteRingStore;
use crate::field::FieldStore;
use crate::homomorphism::Homomorphism;
use crate::integer::int_cast;
use crate::integer::BigIntRing;
use crate::matrix::*;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::rational::RationalField;
use crate::ordered::OrderedRingStore;
use crate::rings::zn::ZnRingStore;
use crate::rings::zn::zn_big;
use crate::rings::zn::ZnRing;
use crate::serialization::{DeserializeWithRing, SerializeWithRing};

use serde::{Deserialize, Serialize};
use feanor_serde::dependent_tuple::DeserializeSeedDependentTuple;
use feanor_serde::impl_deserialize_seed_for_dependent_struct;
use feanor_serde::map::DeserializeSeedMapped;
use feanor_serde::newtype_struct::{DeserializeSeedNewtypeStruct, SerializableNewtypeStruct};
use feanor_serde::seq::{DeserializeSeedSeq, SerializableSeq};

use std::alloc::Global;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::replace;
use std::rc::Rc;

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;
const ZZbig: BigIntRing = BigIntRing::RING;

///
/// Represents a subgroup of an [`AbelianGroupBase`] by a set of generators.
/// Supports computing discrete logarithms, i.e. representing a given element
/// as a combination of the generators.
/// 
/// Note that the used algorithms have a worst case complexity of `O(sqrt(ord^n))`
/// where `ord` is the given multiple of the orders of each generator, and `n`
/// is the number of generators. However, if `ord` is smooth, much faster algorithms
/// are used.
/// 
#[stability::unstable(feature = "enable")]
pub struct SubgroupBase<G: AbelianGroupStore> {
    parent: G,
    generators: Vec<GroupEl<G>>,
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
    scaled_generating_sets: Vec<Vec<Vec<GroupEl<G>>>>
}

///
/// [`GroupStore`] of [`SubgroupBase`]
/// 
#[stability::unstable(feature = "enable")]
#[allow(type_alias_bounds)]
pub type Subgroup<G: AbelianGroupStore> = GroupValue<SubgroupBase<G>>;

impl<G: AbelianGroupStore> Subgroup<G> {

    ///
    /// Creates a new [`GeneratingSet`] representing the subgroup generated
    /// by the given generators.
    /// 
    /// The value `order_multiple` should be a multiple of the order of every
    /// generator, including generators that will be added later on via
    /// [`GeneratingSet::add_generator()`].
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new(group: G, order_multiple: El<BigIntRing>, generators: Vec<GroupEl<G>>) -> Self {
        let n = generators.len();
        if n == 0 {
            return GroupValue::from(SubgroupBase {
                parent: group,
                generators: Vec::new(),
                order_multiple: ZZbig.clone_el(&order_multiple),
                order_factorization: factor(ZZbig, order_multiple).into_iter().map(|(p, e)| (int_cast(p, ZZ, ZZbig), e)).collect(),
                scaled_generating_sets: Vec::new(),
                scaled_relation_lattices: Vec::new()
            });
        } else {
            let mut result = Self::new(group, order_multiple, Vec::new());
            for g in generators {
                result = result.add_generator(g);
            }
            return result;
        }
    }

    ///
    /// Returns the group that this group is a subgroup of.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn parent(&self) -> &G {
        self.get_group().parent()
    }

    ///
    /// Returns the order of the subgroup, i.e. the number of elements.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn subgroup_order(&self) -> El<BigIntRing> {
        self.get_group().subgroup_order()
    }

    ///
    /// Returns the stored generating set of the subgroup.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn generators(&self) -> &[GroupEl<G>] {
        self.get_group().generators()
    }

    ///
    /// Adds a generator to this subgroup, returning a new, larger subgroup.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn add_generator(self, new_gen_base: GroupEl<G>) -> Self {
        Self::from(self.into().add_generator(new_gen_base))
    }

    ///
    /// Checks whether the given element of the parent group is contained
    /// in the subgroup.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn contains(&self, element: &GroupEl<G>) -> bool {
        self.get_group().contains(element)
    }

    ///
    /// Writes the given element of the parent group as a combination of the
    /// subgroup generators, if this exists.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn dlog(&self, target: &GroupEl<G>) -> Option<Vec<i64>> {
        self.get_group().dlog(target)
    }
    
    ///
    /// Returns an iterator over all elements of the subgroup.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn enumerate_elements<'a>(&'a self) -> impl use<'a, G> + Clone + Iterator<Item = GroupEl<G>> {
        self.get_group().enumerate_elements()
    }
}

impl<G: AbelianGroupStore> SubgroupBase<G> {

    #[stability::unstable(feature = "enable")]
    pub fn parent(&self) -> &G {
        &self.parent
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
    /// Returns a set of generators of this subgroup.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn generators(&self) -> &[GroupEl<G>] {
        &self.generators
    }

    ///
    /// Extends the generating set by an additional generator, which is likely
    /// to grow the represented subgroup.
    /// 
    /// The new generator must be of order dividing [`GeneratingSet::order_multiple()`].
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn add_generator(self, new_gen_base: GroupEl<G>) -> Self {
        let group = &self.parent;
        assert!(group.is_identity(&group.pow(&new_gen_base, &self.order_multiple)));
        let ZZ_to_ZZbig = ZZbig.can_hom(&ZZ).unwrap();

        let mut scaled_relation_lattices = Vec::new();
        let mut scaled_generating_sets = Vec::new();
        for p_idx in 0..self.order_factorization.len() {
            
            let (p, e) = self.order_factorization[p_idx];
            let p_bigint = ZZ_to_ZZbig.map(p);
            let power = ZZbig.checked_div(
                &self.order_multiple, 
                &ZZbig.pow(ZZbig.clone_el(&p_bigint), e)
            ).unwrap();
            let gens = self.generators.iter().map(|g| group.pow(g, &power)).collect::<Vec<_>>();
            let new_gen = group.pow(&new_gen_base, &power);
            
            let n = self.generators.len();

            let mut main_relation_matrix = OwnedMatrix::zero(n + 1, n + 1, ZZ);
            for i in 0..n {
                for j in 0..n {
                    *main_relation_matrix.at_mut(i, j) = *self.scaled_relation_lattices[p_idx][e].at(i, j);
                }
            }
            *main_relation_matrix.at_mut(n, n) = -ZZ.pow(p, e);
            for k in 0..e {
                if let Some(dlog) = self.padic_dlog(p_idx, e, &group.pow(&new_gen, &ZZbig.pow(ZZbig.clone_el(&p_bigint), k))) {
                    *main_relation_matrix.at_mut(n, n) = -ZZ.pow(p, k);
                    for j in 0..n {
                        *main_relation_matrix.at_mut(n, j) = dlog[j];
                    }
                    break;
                }
            }
            debug_assert!(main_relation_matrix.data().row_iter().all(|row| group.is_identity(
                &(0..n).fold(group.pow(&new_gen, &ZZ_to_ZZbig.map(row[n])), |current, i| group.op(current, group.pow(&gens[i], &ZZ_to_ZZbig.map(row[i]))))
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
                    let scale = ZZbig.pow(ZZbig.clone_el(&p_bigint), e - i - 1);
                    let result = (0..n).fold(group.pow(&new_gen, &ZZ_to_ZZbig.mul_ref_map(&scale, &row[n])), |current, j| 
                        group.op(current, group.pow(&gens[j], &ZZ_to_ZZbig.mul_ref_map(&scale, &row[j])))
                    );
                    debug_assert!(group.is_identity(&group.pow(&result, &p_bigint)));
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
            scaled_relation_lattices: scaled_relation_lattices,
            parent: self.parent
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
    fn padic_dlog(&self, p_idx: usize, e: usize, target: &GroupEl<G>) -> Option<Vec<i64>> {
        let group = &self.parent;
        let ZZ_to_ZZbig = ZZbig.can_hom(&ZZ).unwrap();
        
        let n = self.generators.len();
        if n == 0 {
            return if group.is_identity(target) { Some(Vec::new()) } else { None };
        } else if e == 0 {
            debug_assert!(group.is_identity(target));
            return Some((0..n).map(|_| 0).collect());
        }

        let p = self.order_factorization[p_idx].0;
        debug_assert!(group.is_identity(&group.pow(target, &ZZbig.pow(ZZ_to_ZZbig.map(p), e))));

        let power = ZZbig.checked_div(
            &self.order_multiple, 
            &ZZbig.pow(int_cast(p, ZZbig, ZZ), e)
        ).unwrap();
        let gens = self.generators.iter().map(|g| group.pow(g, &power)).collect::<Vec<_>>();

        // here we use the power-of-`p` map and the fact that `G/H ~ pG` to compute the dlog in `G/H`
        let G_mod_H_dlog = self.padic_dlog(
            p_idx,
            e - 1,
            &group.pow(target, &ZZ_to_ZZbig.map(p))
        )?;
        debug_assert!(group.eq_el(
            &group.pow(target, &ZZ_to_ZZbig.map(p)),
            &(0..n).fold(group.identity(), |current, i| group.op(current, group.pow(&gens[i], &ZZ_to_ZZbig.map(p * G_mod_H_dlog[i]))))
        ));

        // delta is now in H, i.e. is a p-torsion element
        let delta = (0..n).fold(group.clone_el(target), |current, i|
            group.op(current, group.pow(&gens[i], &ZZ_to_ZZbig.map(-G_mod_H_dlog[i])))
        );
        debug_assert!(group.is_identity(&group.pow(&delta, &ZZ_to_ZZbig.map(p))));

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
            &(0..n).fold(group.identity(), |current, i| group.op(current, group.pow(&gens[i], &ZZ_to_ZZbig.map(result[i]))))
        ));

        return Some(result);
    }

    ///
    /// Returns `true` if the given element is contained in this subgroup.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn contains(&self, element: &GroupEl<G>) -> bool {
        self.dlog(element).is_some()
    }

    ///
    /// Computes a discrete logarithm of `target` w.r.t. the stored set
    /// if generators, or `None` if `target` is not in the subgroup generated by
    /// these generators
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn dlog(&self, target: &GroupEl<G>) -> Option<Vec<i64>> {
        let group = &self.parent;
        let ZZ_to_ZZbig = ZZbig.can_hom(&ZZ).unwrap();
        
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
            let padic_dlog = self.padic_dlog(p_idx, e, &group.pow(target, &power))?;
            for j in 0..n {
                current_dlog[j] = inv_crt(current_dlog[j], padic_dlog[j], &current_order[j], &ZZ.pow(p, e), ZZ);
                current_order[j] *= ZZ.pow(p, e);
            }
        }
        debug_assert!(group.eq_el(
            target,
            &(0..n).fold(group.identity(), |current, i| group.op(current, group.pow(&self.generators[i], &ZZ_to_ZZbig.map(current_dlog[i]))))
        ));

        return Some(current_dlog);
    }

    fn padic_rectangular_form<'a>(&'a self, p_idx: usize) -> Vec<(GroupEl<G>, usize)> {
        let group = &self.parent;
        let (p, e) = self.order_factorization[p_idx];
        let power = ZZbig.checked_div(
            &self.order_multiple, 
            &ZZbig.pow(int_cast(p, ZZbig, ZZ), e)
        ).unwrap();
        let n = self.generators.len();

        if n == 0 {
            return Vec::new();
        }

        let Zpne = zn_big::Zn::new(ZZbig, ZZbig.pow(int_cast(p, ZZbig, StaticRing::<i64>::RING), e * n));
        let mod_pne = Zpne.can_hom(&StaticRing::<i64>::RING).unwrap();
        let relation_lattice = self.scaled_relation_lattices[p_idx][e].data();
        let mut relation_lattice_mod_pne = OwnedMatrix::from_fn(relation_lattice.row_count(), relation_lattice.col_count(), |k, l| mod_pne.map(*relation_lattice.at(k, l)));
        let mut generators = self.generators.iter().map(|g| group.pow(g, &power)).collect::<Vec<_>>();
     
        struct TransformGenerators<'a, G: AbelianGroupStore> {
            group: &'a G,
            generators: &'a mut [GroupEl<G>]
        }
        impl<'a, G: AbelianGroupStore> TransformTarget<zn_big::ZnBase<BigIntRing>> for TransformGenerators<'a, G> {
         fn transform<S: Copy + RingStore<Type = zn_big::ZnBase<BigIntRing>>>(&mut self, ring: S, i: usize, j: usize, transform: &[El<zn_big::Zn<BigIntRing>>; 4]) {
                let transform_inv_det = ring.invert(&ring.sub(
                    ring.mul_ref(&transform[0], &transform[3]),
                    ring.mul_ref(&transform[1], &transform[2])
                )).unwrap();
                let inv_transform = [
                    ring.smallest_positive_lift(ring.mul_ref(&transform[3], &transform_inv_det)),
                    ring.smallest_positive_lift(ring.negate(ring.mul_ref(&transform[1], &transform_inv_det))),
                    ring.smallest_positive_lift(ring.negate(ring.mul_ref(&transform[2], &transform_inv_det))),
                    ring.smallest_positive_lift(ring.mul_ref(&transform[0], &transform_inv_det))
                ];
                let new_gens = (
                    self.group.op(self.group.pow(&self.generators[i], &inv_transform[0]), self.group.pow(&self.generators[j], &inv_transform[1])),
                    self.group.op(self.group.pow(&self.generators[i], &inv_transform[2]), self.group.pow(&self.generators[j], &inv_transform[3])),
                );
                self.generators[i] = new_gens.0;
                self.generators[j] = new_gens.1;
            }
        }
     
        pre_smith(
            &Zpne,
            &mut (),
            &mut TransformGenerators { group: group, generators: &mut generators },
            relation_lattice_mod_pne.data_mut()
        );
     
        return generators.into_iter().enumerate().map(|(i, g)| (
            g,
            int_cast(ZZbig.ideal_gen(Zpne.modulus(), &Zpne.smallest_positive_lift(Zpne.clone_el(relation_lattice_mod_pne.at(i, i)))), ZZ, ZZbig) as usize
        )).collect()
    }

    ///
    /// Returns a list (g[i], l[i]) such that every element of the subgroup
    /// can be uniquely written as `prod_i g[i]^k[i]` with `0 <= k[i] < l[i]`.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn rectangular_form<'a>(&'a self) -> Vec<(GroupEl<G>, usize)> {
        (0..self.order_factorization.len()).flat_map(|p_idx| self.padic_rectangular_form(p_idx)).collect()
    }

    ///
    /// Returns an iterator that yields every element contained in the subgroup
    /// exactly once.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn enumerate_elements<'a>(&'a self) -> impl use<'a, G> + Clone + Iterator<Item = GroupEl<G>> {
        let rectangular_form = Rc::new(self.rectangular_form());
        multi_cartesian_product(
            rectangular_form.iter().map(|(_, l)| 0..*l).collect::<Vec<_>>().into_iter(),
            move |pows| pows.iter().enumerate().fold(self.parent().identity(), |current, (i, e)| 
                self.parent().op(current, self.parent().pow(&rectangular_form[i].0, &int_cast(*e as i64, ZZbig, ZZ)))
            ),
            |_, x| *x
        )
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

impl<G: AbelianGroupStore> PartialEq for SubgroupBase<G> {

    fn eq(&self, other: &Self) -> bool {
        self.parent().get_group() == other.parent().get_group() &&
            other.generators().iter().all(|g| self.contains(g)) &&
            self.generators().iter().all(|g| other.contains(g))
    }
}

impl<G: AbelianGroupStore> Debug for SubgroupBase<G> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<")?;
        for g in self.generators() {
            write!(f, "{}, ", self.parent().formatted_el(g))?;
        }
        write!(f, ">")?;
        return Ok(());
    }
}

impl<G: AbelianGroupStore> AbelianGroupBase for SubgroupBase<G> {

    type Element = GroupEl<G>;

    fn clone_el(&self, x: &Self::Element) -> Self::Element {
        self.parent().clone_el(x)
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.parent().eq_el(lhs, rhs)
    }

    fn hash<H: std::hash::Hasher>(&self, x: &Self::Element, hasher: &mut H) {
        self.parent().hash(x, hasher)
    }
    
    fn identity(&self) -> Self::Element {
        self.parent().identity()
    }

    fn inv(&self, x: &Self::Element) -> Self::Element {
        self.parent().inv(x)
    }

    fn is_identity(&self, x: &Self::Element) -> bool {
        self.parent().is_identity(x)
    }

    fn op(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.parent().op(lhs, rhs)
    }

    fn pow(&self, x: &Self::Element, e: &El<BigIntRing>) -> Self::Element {
        self.parent().pow(x, e)
    }

    fn op_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.parent().op_ref(lhs, rhs)
    }

    fn op_ref_snd(&self, lhs:Self::Element, rhs: &Self::Element) -> Self::Element {
        self.parent().op_ref_snd(lhs, rhs)
    }

    fn fmt_el<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.parent().get_group().fmt_el(value, out)
    }
}

impl<G: AbelianGroupStore + Serialize> Serialize for SubgroupBase<G>
    where G::Type: SerializableElementGroup
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        #[derive(Serialize)]
        struct SubgroupData<'a, Gens: Serialize> {
            order_multiple: SerializeWithRing<'a, BigIntRing>,
            generators: Gens,
            group: ()
        }
        SerializableNewtypeStruct::new("Subgroup", (self.parent(), SubgroupData {
            order_multiple: SerializeWithRing::new(&self.order_multiple, ZZbig), 
            generators: SerializableSeq::new(self.generators.iter().map(|g| SerializeWithGroup::new(g, self.parent()))),
            group: ()
        })).serialize(serializer)
    }
}

impl<'de, G: AbelianGroupStore + Clone + Deserialize<'de>> Deserialize<'de> for SubgroupBase<G>
    where G::Type: SerializableElementGroup
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where  D: serde::Deserializer<'de>
    {
        use serde::de::DeserializeSeed;

        struct DeserializeSeedSubgroupData<G: AbelianGroupStore>
            where G::Type: SerializableElementGroup
        {
            group: G
        }

        impl_deserialize_seed_for_dependent_struct!{
            <{ 'de, G }> pub struct SubgroupData<{'de, G}> using DeserializeSeedSubgroupData<G> {
                order_multiple: El<BigIntRing>: |_| DeserializeWithRing::new(ZZbig),
                generators: Vec<GroupEl<G>>: |master: &DeserializeSeedSubgroupData<G>| {
                    let group_clone = master.group.clone();
                    DeserializeSeedSeq::new((0..).map(move |_| DeserializeWithGroup::new(group_clone.clone())), Vec::new(), |mut current, next| { current.push(next); current })
                },
                group: G: |master: &DeserializeSeedSubgroupData<G>| {
                    let group_clone = master.group.clone();
                    DeserializeSeedMapped::new(PhantomData::<()>, move |()| group_clone)
                }
            } where G: AbelianGroupStore + Clone, G::Type: SerializableElementGroup
        }

        DeserializeSeedNewtypeStruct::new("Subgroup", DeserializeSeedDependentTuple::new(
            PhantomData::<G>,
            |group| DeserializeSeedSubgroupData { group }
        )).deserialize(deserializer).map(|data| Subgroup::new(data.group, data.order_multiple, data.generators).into())
    }
}

impl<G: AbelianGroupStore> Clone for SubgroupBase<G>
    where G: Clone
{
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            generators: self.generators.iter().map(|g| self.parent.clone_el(g)).collect(),
            order_factorization: self.order_factorization.clone(),
            order_multiple: self.order_multiple.clone(),
            scaled_generating_sets: self.scaled_generating_sets.iter().map(|sets| sets.iter().map(|set| set.iter().map(|g| self.parent.clone_el(g)).collect()).collect()).collect(),
            scaled_relation_lattices: self.scaled_relation_lattices.iter().map(|x| x.iter().map(|x| x.clone_matrix(StaticRing::<i64>::RING)).collect()).collect()
        }
    }
}

impl<R> Subgroup<MultGroup<R>>
    where R: RingStore,
        R::Type: ZnRing + HashableElRing + DivisibilityRing
{
    #[stability::unstable(feature = "enable")]
    pub fn for_zn(group: MultGroup<R>, generators: Vec<GroupEl<MultGroup<R>>>) -> Self {
        let n = generators.len();
        if n == 0 {
            let n_factorization = factor(ZZbig, group.underlying_ring().size(ZZbig).unwrap());
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
            return Self::from(SubgroupBase {
                parent: group,
                generators: Vec::new(),
                order_multiple: order,
                order_factorization: order_factorization,
                scaled_generating_sets: Vec::new(),
                scaled_relation_lattices: Vec::new()
            });
        } else {
            let mut result = Self::for_zn(group, Vec::new());
            for g in generators {
                result = result.add_generator(g);
            }
            return result;
        }
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
/// # use feanor_math::group::*;
/// # use feanor_math::primitive_int::*;
/// let group = AddGroup::new(StaticRing::<i64>::RING);
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
/// # use feanor_math::group::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::primitive_int::StaticRing;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::wrapper::*;
/// # use feanor_math::algorithms::discrete_log::*;
/// let ring = Zn::new(17);
/// let group = MultGroup::new(ring);
/// let x = group.from_ring_el(ring.int_hom().map(9)).unwrap();
/// assert_eq!(Some(vec![3]), baby_giant_step(&group, group.pow(&x, &int_cast(3, BigIntRing::RING, StaticRing::<i64>::RING)), &[x], &[BigIntRing::RING.power_of_two(4)]));
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn baby_giant_step<G>(group: G, value: GroupEl<G>, generators: &[GroupEl<G>], dlog_bounds: &[El<BigIntRing>]) -> Option<Vec<i64>> 
    where G: AbelianGroupStore
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
            _ = baby_step_table.insert(HashableGroupEl::new(&group, group.clone_el(&current_els[n - 1])), idx);

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
            current_els[i] = group.op_ref_snd(replace(&mut current_els[i], group.identity()), &generators[i]);
            for j in (i + 1)..n {
                current_els[j] = group.clone_el(&current_els[i]);
            }
        }
    }

    let giant_steps = generators.iter().zip(ns.iter()).map(|(g, n)| group.pow(g, &int_cast(*n, ZZbig, ZZ))).collect::<Vec<_>>();
    // iterate through giant steps
    {
        let start_el = giant_steps.iter().fold(group.identity(), |x, y| group.op_ref_snd(x, y));
        let mut current_els = (0..n).map(|_| group.clone_el(&start_el)).collect::<Vec<_>>();
        let mut current_idxs = (0..n).map(|_| 1).collect::<Vec<_>>();
        for idx in 0..count {
            if let Some(bs_idx) = baby_step_table.get(&HashableGroupEl::new(&group, group.clone_el(&current_els[n - 1]))) {
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
            current_els[i] = group.op_ref_snd(replace(&mut current_els[i], group.identity()), &giant_steps[i]);
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
    let group = MultGroup::new(Zn);
    let generators = vec![group.from_ring_el(base).unwrap()];
    let subgroup = Subgroup::for_zn(group, generators);
    return subgroup.dlog(&subgroup.parent().from_ring_el(value).unwrap())
        .map(|res| res[0]);
}

///
/// Computes the multiplicative order in the group `(Z/nZ)*`.
/// 
pub fn multiplicative_order<R: RingStore>(x: El<R>, Zn: R) -> i64
    where R::Type: ZnRing + HashableElRing
{
    let group = MultGroup::new(Zn);
    let gen_set = Subgroup::for_zn(group, Vec::new());
    let Zn = gen_set.parent().underlying_ring();

    let mut result = ZZbig.one();
    for (p, e) in &gen_set.get_group().order_factorization {
        let mut current = Zn.pow_gen(
            Zn.clone_el(&x), 
            &ZZbig.checked_div(&gen_set.get_group().order_multiple, &ZZbig.pow(int_cast(*p, ZZbig, ZZ), *e)).unwrap(), 
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
struct ProdGroupBase<G: AbelianGroupStore, const N: usize>(G);

#[cfg(test)]
impl<G: AbelianGroupStore, const N: usize> PartialEq for ProdGroupBase<G, N> {
    
    fn eq(&self, other: &Self) -> bool {
        self.0.get_group() == other.0.get_group()
    }
}

#[cfg(test)]
impl<G: AbelianGroupStore, const N: usize> Debug for ProdGroupBase<G, N> {
    
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?})^{}", self.0.get_group(), N)
    }
}

#[cfg(test)]
impl<G: AbelianGroupStore, const N: usize> AbelianGroupBase for ProdGroupBase<G, N> {
    type Element = [GroupEl<G>; N];

    fn clone_el(&self, x: &Self::Element) -> Self::Element {
        from_fn(|i| self.0.clone_el(&x[i]))
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        (0..N).all(|i| self.0.eq_el(&lhs[i], &rhs[i]))
    }

    fn op(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        from_fn(|i| self.0.op_ref(&lhs[i], &rhs[i]))
    }
    
    fn hash<H: std::hash::Hasher>(&self, x: &Self::Element, hasher: &mut H) {
        for i in 0..N {
            self.0.hash(&x[i], hasher)
        }
    }

    fn inv(&self, x: &Self::Element) -> Self::Element {
        from_fn(|i| self.0.inv(&x[i]))
    }

    fn identity(&self) -> Self::Element {
        from_fn(|_| self.0.identity())
    }

    fn fmt_el<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        let mut seq = out.debug_list();
        for x in value {
            _ = seq.entry(&self.0.formatted_el(x));
        }
        return seq.finish();
    }
}

#[cfg(test)]
use crate::rings::zn::zn_static::Zn;
#[cfg(test)]
use oorandom::Rand64;
#[cfg(test)]
use crate::algorithms::matmul::ComputeInnerProduct;
#[cfg(test)]
use crate::group::AddGroup;
#[cfg(test)]
use crate::assert_matrix_eq;
#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;
#[cfg(test)]
use std::array::from_fn;

#[test]
fn test_baby_giant_step() {
    for base_bound in [21, 26, 31, 37] {
        let dlog_bound = [int_cast(base_bound, ZZbig, ZZ)];
        let G = AddGroup::new(ZZ);
        assert_eq!(
            Some(vec![6]), 
            baby_giant_step(&G, 6, &[1], &dlog_bound)
        );
        assert_eq!(
            None, 
            baby_giant_step(&G, 0, &[1], &dlog_bound)
        );

        let G = AddGroup::new(Zn::<20>::RING);
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
    }
    
    let G = AddGroup::new(ZZ);

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
    let G = AddGroup::new(Zn::<81>::RING);

    let subgroup = Subgroup::new(&G, int_cast(81, ZZbig, ZZ), vec![1]);
    assert_matrix_eq!(ZZ, [[-81]], subgroup.get_group().scaled_relation_lattices[0][4]);
    assert_matrix_eq!(ZZ, [[-27]], subgroup.get_group().scaled_relation_lattices[0][3]);
    assert_matrix_eq!(ZZ, [[-9]], subgroup.get_group().scaled_relation_lattices[0][2]);
    assert_matrix_eq!(ZZ, [[-3]], subgroup.get_group().scaled_relation_lattices[0][1]);
    assert_matrix_eq!(ZZ, [[1]], subgroup.get_group().scaled_relation_lattices[0][0]);

    let subgroup = Subgroup::new(&G, int_cast(81, ZZbig, ZZ), vec![3, 6]);
    let matrix = &subgroup.get_group().scaled_relation_lattices[0][4];
    assert_eq!(-27, *matrix.at(0, 0));
    assert_eq!(-1, *matrix.at(1, 1));
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([3, 6].iter())) % 81);

    let subgroup = Subgroup::new(&G, int_cast(81, ZZbig, ZZ), vec![3, 9]);
    let matrix = &subgroup.get_group().scaled_relation_lattices[0][4];
    assert_eq!(-27, *matrix.at(0, 0));
    assert_eq!(-1, *matrix.at(1, 1));
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([3, 9].iter())) % 81);

    let subgroup = Subgroup::new(&G, int_cast(81, ZZbig, ZZ), vec![6, 18, 9]);
    let matrix = &subgroup.get_group().scaled_relation_lattices[0][4];
    assert_eq!(-27, *matrix.at(0, 0));
    assert_eq!(-1, *matrix.at(1, 1));
    assert_eq!(-1, *matrix.at(2, 2));
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([6, 18, 9].iter())) % 81);
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(2).iter().zip([6, 18, 9].iter())) % 81);

    let G = GroupValue::from(ProdGroupBase(AddGroup::new(Zn::<81>::RING)));

    let subgroup = Subgroup::new(&G, int_cast(81, ZZbig, ZZ), vec![[1, 4], [1, 1]]);
    let matrix = &subgroup.get_group().scaled_relation_lattices[0][4];
    assert_eq!(-81, *matrix.at(0, 0));
    assert_eq!(-27, *matrix.at(1, 1));
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([1, 1].iter())) % 81);
    assert_eq!(0, ZZ.get_ring().inner_product_ref(matrix.data().row_at(1).iter().zip([4, 1].iter())) % 81);

    let G = GroupValue::from(ProdGroupBase(AddGroup::new(Zn::<8>::RING)));

    let subgroup = Subgroup::new(&G, int_cast(8, ZZbig, ZZ), vec![[6, 3, 5], [6, 2, 6], [4, 5, 7]]);
    let matrix = &subgroup.get_group().scaled_relation_lattices[0][3];
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
    let G = GroupValue::from(ProdGroupBase(AddGroup::new(ring)));
    let rand_gs = |rng: &mut Rand64| from_fn::<_, 3, _>(|_| ring.random_element(|| rng.rand_u64()));

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let gs = from_fn::<_, 3, _>(|_| rand_gs(&mut rng));
        let subgroup = Subgroup::new(&G, int_cast(1400, ZZbig, ZZ), gs.into());

        let coeffs = rand_gs(&mut rng);
        let val = (0..3).fold(G.identity(), |current, i| G.op(current, G.pow(&gs[i], &int_cast(coeffs[i] as i64, ZZbig, ZZ))));
        let dlog = subgroup.dlog(&val);
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
        let subgroup = Subgroup::new(&G, int_cast(1400, ZZbig, ZZ), gs.into());

        let val = rand_gs(&mut rng);
        let dlog = subgroup.dlog(&val);
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
                assert!(solved == crate::algorithms::linsolve::SolveResult::NoSolution);
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
    let group = MultGroup::new(ring);
    let g1 = group.from_ring_el(ring.int_hom().map(2)).unwrap();
    let g2 = group.from_ring_el(ring.int_hom().map(37)).unwrap();

    let mut subgroup = Subgroup::for_zn(group.clone(), Vec::new());
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(1), subgroup.subgroup_order());

    let next_gen = subgroup.parent().clone_el(&g1);
    subgroup = subgroup.add_generator(next_gen);
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(24), subgroup.subgroup_order());
    
    let next_gen = subgroup.parent().pow(&g1, &ZZbig.int_hom().map(2));
    subgroup = subgroup.add_generator(next_gen);
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(24), subgroup.subgroup_order());
    
    let next_gen = subgroup.parent().clone_el(&g2);
    subgroup = subgroup.add_generator(next_gen);
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(96), subgroup.subgroup_order());
    
    let generating_set = Subgroup::for_zn(group, vec![g2]);
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(16), generating_set.subgroup_order());

}

#[test]
fn test_enumerate_elements() {
    let ring = Zn::<45>::RING;
    let group = AddGroup::new(ring);

    assert_eq!(vec![ring.zero()], Subgroup::new(group.clone(), int_cast(45, ZZbig, ZZ), Vec::new()).enumerate_elements().collect::<Vec<_>>());

    let subgroup = Subgroup::new(group, int_cast(45, ZZbig, ZZ), vec![9, 15]);
    let mut elements = subgroup.enumerate_elements().collect::<Vec<_>>();
    elements.sort_unstable();
    assert_eq!((0..45).step_by(3).collect::<Vec<_>>(), elements);
}