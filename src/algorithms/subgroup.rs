use std::alloc::Global;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::replace;
use std::rc::Rc;
use std::sync::OnceLock;

use feanor_serde::dependent_tuple::DeserializeSeedDependentTuple;
use feanor_serde::impl_deserialize_seed_for_dependent_struct;
use feanor_serde::map::DeserializeSeedMapped;
use feanor_serde::newtype_struct::{DeserializeSeedNewtypeStruct, SerializableNewtypeStruct};
use feanor_serde::seq::{DeserializeSeedSeq, SerializableSeq};
use oorandom::Rand64;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::PROBABILISTIC_REPETITIONS;
use crate::algorithms::int_bisect::root_floor;
use crate::algorithms::int_factor::factor;
use crate::algorithms::lattices::*;
use crate::algorithms::linsolve::smith::{determinant_using_pre_smith, pre_smith};
use crate::algorithms::matmul::{MatmulAlgorithm, STANDARD_MATMUL};
use crate::group::{HashableGroupEl, MultGroup, *};
use crate::homomorphism::Homomorphism;
use crate::iters::multi_cartesian_product;
use crate::matrix::transform::*;
use crate::matrix::*;
use crate::prelude::*;
use crate::ring::HashableElRing;
use crate::ring_impls::zn::zn_big::*;
use crate::ring_impls::zn::zn_rns::ZnRNS;
use crate::ring_impls::zn::{ZnRing, ZnRingStore};
use crate::ring_properties::divisibility::{DivisibilityRing, DivisibilityRingStore};
use crate::ring_properties::finite::FiniteRingStore;
use crate::ring_properties::integer::{BigIntRing, int_cast, int_range_exclusive};
use crate::ring_properties::ordered::OrderedRingStore;
use crate::ring_properties::pid::PrincipalIdealRingStore;
use crate::ring_properties::serialization::{DeserializeWithRing, SerializeWithRing};
use crate::seq::VectorView;

/// Represents a subgroup of an [`AbelianGroupBase`] by a set of generators.
/// Supports computing discrete logarithms, i.e. representing a given element
/// as a combination of the generators.
///
/// Note that the used algorithms have a worst case complexity of `O(sqrt(ord^n))`
/// where `ord` is the given multiple of the orders of each generator, and `n`
/// is the number of generators. However, if `ord` is smooth, much faster algorithms
/// are used.
pub struct SubgroupBase<G: AbelianGroupStore> {
    parent: G,
    generators: Vec<GEl<G>>,
    order_multiple: El<BigIntRing>,
    /// factorization of [`SubgroupBase::order_multiple`]
    order_factorization: Vec<(El<BigIntRing>, usize)>,
    /// the `(i, j)`-th entry has columns that form a basis of the relation lattice of
    /// the set `n/pi^j g1, ..., n/pi^j gk` (where `n` is [`SubgroupBase::order_multiple`],
    /// and the `pi^ei` are its prime power factors).
    padic_relation_lattices: Vec<Vec<OwnedMatrix<El<BigIntRing>>>>,
    /// the `(i, j, k)`-th entry contains `sum_l col[l] n/pi^(j + 1) gl`, where
    /// `col` is the `k`-th column of `scaled_relation_lattice[i, j]`; These values
    /// are important, since they form a basis of the `p`-torsion subgroup of
    /// `< n/pi^(j + 1) g1, ..., n/pi^(j + 1) gk >`
    padic_generating_sets: Vec<Vec<Vec<GEl<G>>>>,
    subgroup_order: OnceLock<El<BigIntRing>>,
    global_relation_lattice: OnceLock<OwnedMatrix<El<BigIntRing>>>,
    cyclic_decomposition: OnceLock<Vec<(GEl<G>, El<BigIntRing>)>>,
}

/// [`GroupStore`] of [`SubgroupBase`]
pub type Subgroup<G> = GroupValue<SubgroupBase<G>>;

impl<G: AbelianGroupStore> Subgroup<G> {
    /// Creates a new [`GeneratingSet`] representing the subgroup generated
    /// by the given generators.
    ///
    /// The value `order_multiple` should be a multiple of the order of every given
    /// generator.
    #[instrument(skip_all, level = "trace")]
    pub fn new(group: G, order_multiple: El<BigIntRing>, generators: Vec<GEl<G>>) -> Self {
        let n = generators.len();
        if n == 0 {
            return GroupValue::from(SubgroupBase {
                parent: group,
                generators: Vec::new(),
                order_multiple: order_multiple.clone(),
                order_factorization: factor(ZZbig, order_multiple),
                padic_generating_sets: Vec::new(),
                padic_relation_lattices: Vec::new(),
                global_relation_lattice: OnceLock::new(),
                cyclic_decomposition: OnceLock::new(),
                subgroup_order: OnceLock::new(),
            });
        } else {
            let mut result = Self::new(group, order_multiple.clone(), Vec::new());
            for g in generators {
                result = result.add_generator(g, &order_multiple);
            }
            return result;
        }
    }

    /// Returns the group that this group is a subgroup of.
    pub fn parent(&self) -> &G { self.get_group().parent() }

    /// Returns the order of the subgroup, i.e. the number of elements.
    pub fn subgroup_order(&self) -> &El<BigIntRing> { self.get_group().subgroup_order() }

    /// Returns the stored generating set of the subgroup.
    pub fn generators(&self) -> &[GEl<G>] { self.get_group().generators() }

    /// Returns a multiple of the order of each element in the subgroup
    /// generated by this generating set.
    #[stability::unstable(feature = "enable")]
    pub fn order_multiple(&self) -> &El<BigIntRing> { self.get_group().order_multiple() }

    /// Adds a generator to this subgroup, returning a new, larger subgroup.
    pub fn add_generator(self, generator: GEl<G>, generator_order_multiple: &El<BigIntRing>) -> Self {
        Self::from(self.into().add_generator(generator, generator_order_multiple))
    }

    /// Returns (g1, n1), ..., (gr, nr) such that we have the isomorphism
    /// ```text
    ///   G -> C_n1 x ... x C_nr,    gi -> ei
    /// ```
    pub fn cyclic_decomposition(&self) -> &[(GEl<Self>, El<BigIntRing>)]
    where
        G: Clone,
    {
        self.get_group().cyclic_decomposition()
    }

    /// Checks whether the given element of the parent group is contained
    /// in the subgroup.
    pub fn contains(&self, element: &GEl<G>) -> bool { self.get_group().contains(element) }

    /// Writes the given element of the parent group as a combination of the
    /// subgroup generators, if this exists.
    pub fn dlog(&self, target: &GEl<G>) -> Option<Vec<El<BigIntRing>>> { self.get_group().dlog(target) }

    /// Returns an iterator over all elements of the subgroup.
    pub fn enumerate_elements<'a>(&'a self) -> impl use<'a, G> + Clone + Iterator<Item = GEl<G>>
    where
        G: Clone,
    {
        self.get_group().enumerate_elements()
    }

    pub fn relation_lattice(&self) -> &OwnedMatrix<El<BigIntRing>> { self.get_group().relation_lattice() }

    /// Returns (g1, n1), ..., (gr, nr) such that we have the isomorphism
    /// ```text
    ///   C_n1 x ... x C_nr -> G / H,    ei -> gi mod H
    /// ```
    /// Requires that H is contained in this subgroup
    pub fn quotient_cyclic_decomposition<S>(&self, H: &Subgroup<S>) -> Vec<(GEl<G>, El<BigIntRing>)>
    where
        S: AbelianGroupStore<Group = G::Group>,
        G: Clone,
    {
        self.get_group().quotient_cyclic_decomposition(H.get_group())
    }

    pub fn sum<S>(&self, other: &Subgroup<S>) -> Subgroup<G>
    where
        S: AbelianGroupStore<Group = G::Group>,
        G: Clone,
    {
        Subgroup::from(self.get_group().sum(other.get_group()))
    }

    pub fn intersection<S>(&self, other: &Subgroup<S>) -> Subgroup<G>
    where
        S: AbelianGroupStore<Group = G::Group>,
        G: Clone,
    {
        Subgroup::from(self.get_group().intersection(other.get_group()))
    }
}

impl<G: AbelianGroupStore> SubgroupBase<G> {
    pub fn parent(&self) -> &G { &self.parent }

    /// The number of elements in the subgroup generated by this generating set.
    #[instrument(skip_all, level = "trace")]
    pub fn subgroup_order(&self) -> &El<BigIntRing> {
        self.subgroup_order.get_or_init(|| {
            let mut result = ZZbig.one();
            let n = self.generators.len();
            if n == 0 {
                return result;
            }
            for i in 0..self.order_factorization.len() {
                let (p, e) = self.order_factorization[i].clone();
                let relation_lattice = self.padic_relation_lattices[i][e].data();
                let ring = ZnGB::new(ZZbig, ZZbig.pow(p, e * n + 1));
                let to_ring = ring.can_hom(&ZZbig).unwrap();
                let mut A = OwnedMatrix::from_fn(relation_lattice.row_count(), relation_lattice.col_count(), |k, l| {
                    to_ring.map_ref(relation_lattice.at(k, l))
                });
                let relation_lattice_det = determinant_using_pre_smith(&ring, A.data_mut(), Global);
                ZZbig.mul_assign(
                    &mut result,
                    ZZbig.ideal_gen(ring.modulus(), &ring.smallest_positive_lift(relation_lattice_det)),
                );
            }
            return result;
        })
    }

    /// Returns a set of generators of this subgroup.
    pub fn generators(&self) -> &[GEl<G>] { &self.generators }

    #[instrument(skip_all, level = "trace")]
    pub fn relation_lattice(&self) -> &OwnedMatrix<El<BigIntRing>> {
        self.global_relation_lattice.get_or_init(|| {
            let Zn = ZnRNS::new(
                self.order_factorization
                    .iter()
                    .map(|(p, e)| ZnGB::new(ZZbig, ZZbig.pow(p.clone(), *e)))
                    .collect(),
                ZZbig,
            );
            let k = self.generators.len();
            let mut result = OwnedMatrix::zero(k, k * self.order_factorization.len(), ZZbig);
            for (idx, (Zp, p_relations)) in Zn.as_iter().zip(self.padic_relation_lattices.iter()).enumerate() {
                let p_relations = p_relations.last().unwrap();
                let crt_unit_vector = Zn.smallest_lift(Zn.from_congruence(Zn.as_iter().map(|Zp_| {
                    if Zp_.get_ring() == Zp.get_ring() {
                        Zp_.one()
                    } else {
                        Zp_.zero()
                    }
                })));
                for i in 0..k {
                    for j in 0..k {
                        *result.at_mut(i, j + idx * k) =
                            ZZbig.mul_ref_fst(&crt_unit_vector, p_relations.at(i, j).clone());
                    }
                }
            }
            return lattice_basis_from_generating_set(ZZbig, result, None);
        })
    }

    #[instrument(skip_all, level = "trace")]
    pub fn sum<S>(&self, other: &SubgroupBase<S>) -> Subgroup<G>
    where
        S: AbelianGroupStore<Group = G::Group>,
        G: Clone,
    {
        assert!(self.parent().get_group() == other.parent().get_group());
        let mut result = self.clone();
        for g in &other.generators {
            result = result.add_generator(g.clone(), other.order_multiple());
        }
        return GroupValue::from(result);
    }

    #[instrument(skip_all, level = "trace")]
    pub fn intersection<S>(&self, other: &SubgroupBase<S>) -> Subgroup<G>
    where
        S: AbelianGroupStore<Group = G::Group>,
        G: Clone,
    {
        let sum = self.sum(other);
        let group = self.parent();
        debug_assert!(
            sum.generators()
                .iter()
                .zip(&self.generators)
                .all(|(l, r)| group.eq_el(l, r))
        );
        debug_assert!(
            sum.generators()
                .iter()
                .skip(self.generators.len())
                .zip(&other.generators)
                .all(|(l, r)| group.eq_el(l, r))
        );
        let A = sum
            .get_group()
            .relation_lattice()
            .data()
            .restrict_rows(0..self.generators.len());
        let intersection_generators = A
            .col_iter()
            .map(|col| {
                col.as_iter()
                    .zip(self.generators())
                    .map(|(pow, g)| group.pow_bigint(g.clone(), pow))
                    .fold(group.identity(), |x, y| group.op(x, y))
            })
            .collect::<Vec<_>>();
        // TODO: we can actually compute the the relation lattice of the result here, as the pullback
        // of the relation lattice of self under left-multiplication by A
        return Subgroup::new(
            sum.into().parent,
            ZZbig.gcd(&self.order_multiple, &other.order_multiple),
            intersection_generators,
        );
    }

    /// Returns `true` if the given element is contained in this subgroup.
    pub fn contains(&self, element: &GEl<G>) -> bool { self.dlog(element).is_some() }

    /// Computes a discrete logarithm of `target` w.r.t. the stored set
    /// if generators, or `None` if `target` is not in the subgroup generated by
    /// these generators
    #[instrument(skip_all, level = "trace")]
    pub fn dlog(&self, target: &GEl<G>) -> Option<Vec<El<BigIntRing>>> {
        let group = &self.parent;
        if !group.is_identity(&group.pow_bigint(target.clone(), &self.order_multiple)) {
            return None;
        }

        let n = self.generators.len();
        if n == 0 {
            return if group.is_identity(target) {
                Some(Vec::new())
            } else {
                None
            };
        }

        let mut current_dlog = (0..n).map(|_| ZZbig.zero()).collect::<Vec<_>>();
        let mut current_order = (0..n).map(|_| ZZbig.one()).collect::<Vec<_>>();

        for p_idx in 0..self.order_factorization.len() {
            let (p, e) = &self.order_factorization[p_idx];
            let pe = ZZbig.pow(p.clone(), *e);
            let power = ZZbig.checked_div(&self.order_multiple, &pe).unwrap();
            let padic_dlog = self.padic_dlog(p_idx, *e, &group.pow_bigint(target.clone(), &power))?;
            for j in 0..n {
                current_dlog[j] = ZZbig.inv_crt([&current_dlog[j], &padic_dlog[j]], [&current_order[j], &pe]);
                ZZbig.mul_assign_ref(&mut current_order[j], &pe);
                if ZZbig.is_neg(&current_dlog[j]) {
                    ZZbig.add_assign_ref(&mut current_dlog[j], &current_order[j]);
                }
            }
        }
        debug_assert!(group.eq_el(
            target,
            &(0..n).fold(group.identity(), |current, i| {
                group.op(current, group.pow_bigint(self.generators[i].clone(), &current_dlog[i]))
            })
        ));

        return Some(current_dlog);
    }

    /// Returns an iterator that yields every element contained in the subgroup
    /// exactly once.
    pub fn enumerate_elements<'a>(&'a self) -> impl use<'a, G> + Clone + Iterator<Item = GEl<G>>
    where
        G: Clone,
    {
        let rectangular_form = Rc::new(self.cyclic_decomposition());
        multi_cartesian_product(
            rectangular_form
                .iter()
                .map(|(_, l)| int_range_exclusive(ZZbig, l.clone()))
                .collect::<Vec<_>>()
                .into_iter(),
            move |pows| {
                pows.iter()
                    .enumerate()
                    .fold(self.parent().identity(), |current, (i, e)| {
                        self.parent()
                            .op(current, self.parent().pow_bigint(rectangular_form[i].0.clone(), e))
                    })
            },
            |_, x| x.clone(),
        )
    }

    /// Returns a multiple of the order of each element in the subgroup
    /// generated by this generating set.
    #[stability::unstable(feature = "enable")]
    pub fn order_multiple(&self) -> &El<BigIntRing> { &self.order_multiple }

    fn increase_order_multiple(&mut self, new_order_multiple: El<BigIntRing>) {
        assert!(ZZbig.divides(&new_order_multiple, &self.order_multiple));
        let k = self.generators().len();
        let new_factorization = factor(ZZbig, new_order_multiple.clone());

        let mut new_scaled_relation_lattices: Vec<Vec<_>> = Vec::new();
        let mut new_scaled_generating_sets: Vec<Vec<Vec<_>>> = Vec::new();
        for (p, e) in &new_factorization {
            if let Some((idx_old, (_, e_old))) = self
                .order_factorization
                .iter()
                .enumerate()
                .filter(|(_, (p_, _))| ZZbig.eq_el(p_, p))
                .next()
            {
                debug_assert!(e >= e_old);
                new_scaled_relation_lattices.push(
                    (0..(e - e_old))
                        .map(|_| OwnedMatrix::identity(k, k, ZZbig))
                        .chain(self.padic_relation_lattices[idx_old].drain(..))
                        .collect(),
                );
            } else {
                new_scaled_relation_lattices.push((0..=*e).map(|_| OwnedMatrix::identity(k, k, ZZbig)).collect());
            }
            let power = ZZbig
                .checked_div(&new_order_multiple, &ZZbig.pow(p.clone(), *e))
                .unwrap();
            let gens = self
                .generators
                .iter()
                .map(|g| self.parent().pow_bigint(g.clone(), &power))
                .collect::<Vec<_>>();
            new_scaled_generating_sets.push(Self::compute_scaled_generating_set(
                self.parent(),
                p,
                *e,
                gens.iter().cloned(),
                new_scaled_relation_lattices.last().unwrap(),
            ));
        }

        self.order_factorization = new_factorization;
        self.order_multiple = new_order_multiple;
        self.padic_generating_sets = new_scaled_generating_sets;
        self.padic_relation_lattices = new_scaled_relation_lattices;
    }

    fn compute_scaled_generating_set<I>(
        group: &G,
        p: &El<BigIntRing>,
        e: usize,
        gens: I,
        scaled_relation_lattices: &[OwnedMatrix<El<BigIntRing>>],
    ) -> Vec<Vec<GEl<G>>>
    where
        I: ExactSizeIterator<Item = GEl<G>> + Clone,
    {
        let mut generating_sets = Vec::new();
        let k = gens.len();
        assert_eq!(k, scaled_relation_lattices[0].row_count());
        assert_eq!(k, scaled_relation_lattices[0].col_count());
        for i in 0..e {
            let generating_set = scaled_relation_lattices[i]
                .data()
                .col_iter()
                .map(|col| {
                    let scale = ZZbig.pow(p.clone(), e - i - 1);
                    let result = gens
                        .clone()
                        .zip(col.as_iter())
                        .fold(group.identity(), |current, (g, pow)| {
                            group.op(current, group.pow_bigint(g, &ZZbig.mul_ref(&scale, pow)))
                        });
                    debug_assert!(group.is_identity(&group.pow_bigint(result.clone(), p)));
                    result
                })
                .collect::<Vec<_>>();
            generating_sets.push(generating_set);
        }
        return generating_sets;
    }

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
    #[instrument(skip_all, level = "trace")]
    fn padic_dlog(&self, p_idx: usize, e: usize, target: &GEl<G>) -> Option<Vec<El<BigIntRing>>> {
        let group = &self.parent;

        let n = self.generators.len();
        if n == 0 {
            return if group.is_identity(target) {
                Some(Vec::new())
            } else {
                None
            };
        } else if e == 0 {
            debug_assert!(group.is_identity(target));
            return Some((0..n).map(|_| ZZbig.zero()).collect());
        }

        let p = &self.order_factorization[p_idx].0;
        let pe = ZZbig.pow(p.clone(), e);
        debug_assert!(group.is_identity(&group.pow_bigint(target.clone(), &pe)));

        let power = ZZbig.checked_div(&self.order_multiple, &pe).unwrap();
        let gens = self
            .generators
            .iter()
            .map(|g| group.pow_bigint(g.clone(), &power))
            .collect::<Vec<_>>();

        // here we use the power-of-`p` map and the fact that `G/H ~ pG` to compute the dlog in `G/H`
        let G_mod_H_dlog = self.padic_dlog(p_idx, e - 1, &group.pow_bigint(target.clone(), &p))?;
        debug_assert!(group.eq_el(
            &group.pow_bigint(target.clone(), &p),
            &(0..n).fold(group.identity(), |current, i| {
                group.op(
                    current,
                    group.pow_bigint(gens[i].clone(), &ZZbig.mul_ref(&p, &G_mod_H_dlog[i])),
                )
            })
        ));

        // delta is now in H, i.e. is a p-torsion element
        let delta = (0..n).fold(target.clone(), |current, i| {
            group.op(
                current,
                group.pow_bigint(gens[i].clone(), &ZZbig.neg(G_mod_H_dlog[i].clone())),
            )
        });
        debug_assert!(group.is_identity(&group.pow_bigint(delta.clone(), &p)));

        let H_generators = &self.padic_generating_sets[p_idx][e - 1];

        let p_i64 = int_cast(p.clone(), ZZi64, ZZbig);
        let H_dlog_wrt_H_gens =
            baby_giant_step(group, delta, &H_generators, &(0..n).map(|_| p_i64).collect::<Vec<_>>())?
                .into_iter()
                .map(|x| int_cast(x, ZZbig, ZZi64))
                .collect::<Vec<_>>();
        let H_dlog = {
            let mut result = (0..n).map(|_| ZZbig.zero()).collect::<Vec<_>>();
            STANDARD_MATMUL.matmul(
                TransposableSubmatrix::from(Submatrix::from_1d(&H_dlog_wrt_H_gens, 1, n)),
                TransposableSubmatrix::from(self.padic_relation_lattices[p_idx][e - 1].data()).transpose(),
                TransposableSubmatrixMut::from(SubmatrixMut::from_1d(&mut result, 1, n)),
                ZZbig,
            );
            result
        };

        let result = G_mod_H_dlog
            .into_iter()
            .zip(H_dlog.into_iter())
            .map(|(x, y)| ZZbig.add(x, y))
            .collect::<Vec<_>>();
        debug_assert!(group.eq_el(
            target,
            &(0..n).fold(group.identity(), |current, i| {
                group.op(current, group.pow_bigint(gens[i].clone(), &result[i]))
            })
        ));

        return Some(result);
    }

    /// Returns (g1, n1), ..., (gr, nr) such that we have the isomorphism
    /// ```text
    ///   C_n1 x ... x C_nr -> G,    ei -> gi
    /// ```
    pub fn cyclic_decomposition(&self) -> &[(GEl<G>, El<BigIntRing>)]
    where
        G: Clone,
    {
        self.cyclic_decomposition.get_or_init(|| {
            self.quotient_cyclic_decomposition(Subgroup::new(&self.parent, ZZbig.one(), Vec::new()).get_group())
        })
    }

    /// Returns (g1, n1), ..., (gr, nr) such that we have the isomorphism
    /// ```text
    ///   C_n1 x ... x C_nr -> G / H,    ei -> gi mod H
    /// ```
    /// Requires that H is contained in this subgroup.
    #[instrument(skip_all, level = "trace")]
    pub fn quotient_cyclic_decomposition<S>(&self, H: &SubgroupBase<S>) -> Vec<(GEl<G>, El<BigIntRing>)>
    where
        S: AbelianGroupStore<Group = G::Group>,
        G: Clone,
    {
        assert!(self.parent().get_group() == H.parent().get_group());
        let group = self.parent();

        // common case: the generators of H are a suffix of the generators of self
        if !self.generators[(self.generators.len() - H.generators.len())..]
            .iter()
            .zip(&H.generators)
            .all(|(x, y)| group.eq_el(x, y))
        {
            let sum_group = self.sum(H);
            assert!(
                ZZbig.eq_el(sum_group.subgroup_order(), self.subgroup_order()),
                "H is not contained in self"
            );
            return sum_group.into().quotient_cyclic_decomposition(H);
        }
        return (0..self.order_factorization.len())
            .flat_map(|p_idx| self.padic_cyclic_decomposition(p_idx, self.generators.len() - H.generators.len()))
            .collect::<Vec<_>>();
    }

    /// Computes a cyclic decomposition of the order-p part of this group, modulo the subgroup
    /// spanned by the last `(generators.len() - project_onto)` generators.
    #[instrument(skip_all, level = "trace")]
    fn padic_cyclic_decomposition<'a>(&'a self, p_idx: usize, project_onto: usize) -> Vec<(GEl<G>, El<BigIntRing>)> {
        let group = &self.parent;
        let (p, e) = &self.order_factorization[p_idx];
        let pe = ZZbig.pow(p.clone(), *e);
        let power = ZZbig.checked_div(&self.order_multiple, &pe).unwrap();
        let n = self.generators.len();

        if n == 0 {
            return Vec::new();
        }
        let Zpe = ZnGB::new(ZZbig, pe);
        let mod_pe = Zpe.can_hom(&ZZbig).unwrap();
        let mut A = self.padic_relation_lattices[p_idx]
            .last()
            .unwrap()
            .data()
            .restrict_rows(0..project_onto)
            .to_owned()
            .map(|x| mod_pe.map(x));
        let mut L_negT = OwnedMatrix::identity(project_onto, project_onto, &Zpe);
        pre_smith(
            &Zpe,
            &mut InvertTransform::new(TransposeTransform::new(TransformRows::new(L_negT.data_mut()))),
            &mut (),
            A.data_mut(),
        );
        let mut diagonal = Vec::new();
        for i in 0..project_onto {
            let pivot = Zpe.smallest_positive_lift(A.at(i, i).clone());
            let order = ZZbig.gcd(&pivot, Zpe.modulus());
            if !ZZbig.is_zero(&pivot) {
                let factor = mod_pe.map(ZZbig.checked_div(&pivot, &order).unwrap());
                diagonal.push(order);
                debug_assert!(Zpe.is_unit(&factor));
                for j in 0..project_onto {
                    Zpe.mul_assign_ref(L_negT.at_mut(i, j), &factor);
                }
            } else {
                diagonal.push(order);
            }
        }
        let row_as_group_el = |row: &[El<ZnGB<BigIntRing>>]| {
            row.iter()
                .zip(self.generators.iter())
                .map(|(c, g)| {
                    group.pow_bigint(
                        g.clone(),
                        &ZZbig.mul_ref_fst(&power, Zpe.smallest_positive_lift(c.clone())),
                    )
                })
                .fold(group.identity(), |x, y| group.op(x, y))
        };
        return L_negT
            .data()
            .row_iter()
            .zip(diagonal)
            .map(|(row, n)| (row_as_group_el(row), n))
            .filter(|(_, n)| ZZbig.is_gt(n, &ZZbig.one()))
            .collect();
    }

    /// Extends the generating set by an additional generator, which is likely
    /// to grow the represented subgroup.
    ///
    /// The new generator must be of order dividing [`GeneratingSet::order_multiple()`].
    #[instrument(skip_all, level = "trace")]
    pub fn add_generator(mut self, new_generator: GEl<G>, new_generator_order_multiple: &El<BigIntRing>) -> Self {
        let group = &self.parent;
        assert!(group.is_identity(&group.pow_bigint(new_generator.clone(), new_generator_order_multiple)));
        if !ZZbig.divides(&self.order_multiple, &new_generator_order_multiple) {
            self.increase_order_multiple(ZZbig.lcm(&self.order_multiple, new_generator_order_multiple));
        }
        let group = &self.parent;

        let mut padic_relation_lattices = Vec::new();
        let mut padic_generating_sets = Vec::new();
        for p_idx in 0..self.order_factorization.len() {
            let (p, e) = &self.order_factorization[p_idx];
            let pe = ZZbig.pow(p.clone(), *e);
            let power = ZZbig.checked_div(&self.order_multiple, &pe).unwrap();
            let gens = self
                .generators
                .iter()
                .map(|g| group.pow_bigint(g.clone(), &power))
                .collect::<Vec<_>>();
            let new_gen = group.pow_bigint(new_generator.clone(), &power);

            let n = self.generators.len();

            let mut main_relation_matrix = OwnedMatrix::zero(n + 1, n + 1, ZZbig);
            for i in 0..n {
                for j in 0..n {
                    *main_relation_matrix.at_mut(i, j) = self.padic_relation_lattices[p_idx][*e].at(i, j).clone();
                }
            }
            *main_relation_matrix.at_mut(n, n) = ZZbig.neg(pe.clone());
            for k in 0..*e {
                if let Some(dlog) =
                    self.padic_dlog(p_idx, *e, &group.pow_bigint(new_gen.clone(), &ZZbig.pow(p.clone(), k)))
                {
                    *main_relation_matrix.at_mut(n, n) = ZZbig.neg(ZZbig.pow(p.clone(), k));
                    for (j, val) in dlog.into_iter().enumerate() {
                        *main_relation_matrix.at_mut(j, n) = val;
                    }
                    break;
                }
            }
            debug_assert!(main_relation_matrix.data().col_iter().all(|col| group.is_identity(
                &(0..n).fold(group.pow_bigint(new_gen.clone(), col.at(n)), |current, i| {
                    group.op(current, group.pow_bigint(gens[i].clone(), col.at(i)))
                })
            )));

            let mut result = Vec::with_capacity(e + 1);
            result.extend(lattice_p_saturation_tower(ZZbig, p.clone(), main_relation_matrix, None));
            debug_assert!(result.len() <= e + 1);
            result.resize_with(e + 1, || OwnedMatrix::identity(n + 1, n + 1, ZZbig));
            result.reverse();
            padic_relation_lattices.push(result);
            padic_generating_sets.push(Self::compute_scaled_generating_set(
                group,
                p,
                *e,
                gens.iter()
                    .chain([&new_gen])
                    .collect::<Vec<_>>()
                    .iter()
                    .copied()
                    .cloned(),
                padic_relation_lattices.last().unwrap(),
            ));
        }

        return Self {
            generators: self
                .generators
                .iter()
                .map(|g| g.clone())
                .chain([new_generator].into_iter())
                .collect(),
            order_multiple: self.order_multiple.clone(),
            order_factorization: self.order_factorization.clone(),
            padic_generating_sets,
            padic_relation_lattices,
            parent: self.parent,
            global_relation_lattice: OnceLock::new(),
            cyclic_decomposition: OnceLock::new(),
            subgroup_order: OnceLock::new(),
        };
    }
}

impl<G: AbelianGroupStore> PartialEq for SubgroupBase<G> {
    fn eq(&self, other: &Self) -> bool {
        self.parent().get_group() == other.parent().get_group()
            && other.generators().iter().all(|g| self.contains(g))
            && self.generators().iter().all(|g| other.contains(g))
    }
}

impl<G: AbelianGroupStore> Debug for SubgroupBase<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<")?;
        let mut gen_iter = self.generators().iter();
        if let Some(g) = gen_iter.next() {
            write!(f, "{}", self.parent().formatted_el(g))?;
        }
        for g in gen_iter {
            write!(f, ", {}", self.parent().formatted_el(g))?;
        }
        write!(f, ">")?;
        return Ok(());
    }
}

impl<G: AbelianGroupStore> AbelianGroupBase for SubgroupBase<G> {
    type Element = GEl<G>;

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool { self.parent().eq_el(lhs, rhs) }
    fn hash<H: std::hash::Hasher>(&self, x: &Self::Element, hasher: &mut H) { self.parent().hash(x, hasher) }
    fn identity(&self) -> Self::Element { self.parent().identity() }
    fn inv(&self, x: &Self::Element) -> Self::Element { self.parent().inv(x) }
    fn is_identity(&self, x: &Self::Element) -> bool { self.parent().is_identity(x) }
    fn op(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element { self.parent().op(lhs, rhs) }
    fn op_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element { self.parent().op_ref(lhs, rhs) }

    fn op_ref_snd(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.parent().op_ref_snd(lhs, rhs)
    }

    fn fmt_el<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.parent().get_group().fmt_el(value, out)
    }

    fn pow_gen<R>(&self, x: Self::Element, power: &El<R>, integers: R) -> Self::Element
    where
        R: RingStore,
        R::Ring: IntegerRing,
    {
        self.parent().pow_gen(x, power, integers)
    }
}

impl<G: AbelianGroupStore + Serialize> Serialize for SubgroupBase<G>
where
    G::Group: SerializableElementGroup,
{
    #[instrument(skip_all, level = "trace")]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct SubgroupData<'a, Gens: Serialize> {
            order_multiple: SerializeWithRing<'a, BigIntRing>,
            generators: Gens,
            group: (),
        }
        SerializableNewtypeStruct::new(
            "Subgroup",
            (
                self.parent(),
                SubgroupData {
                    order_multiple: SerializeWithRing::new(&self.order_multiple, ZZbig),
                    generators: SerializableSeq::new(
                        self.generators
                            .iter()
                            .map(|g| SerializeWithGroup::new(g, self.parent())),
                    ),
                    group: (),
                },
            ),
        )
        .serialize(serializer)
    }
}

impl<'de, G: AbelianGroupStore + Clone + Deserialize<'de>> Deserialize<'de> for SubgroupBase<G>
where
    G::Group: SerializableElementGroup,
{
    #[instrument(skip_all, level = "trace")]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::DeserializeSeed;

        struct DeserializeSeedSubgroupData<G: AbelianGroupStore>
        where
            G::Group: SerializableElementGroup,
        {
            group: G,
        }

        impl_deserialize_seed_for_dependent_struct! {
            <{ 'de, G }> pub struct SubgroupData<{'de, G}> using DeserializeSeedSubgroupData<G> {
                order_multiple: El<BigIntRing>: |_| DeserializeWithRing::new(ZZbig),
                generators: Vec<GEl<G>>: |master: &DeserializeSeedSubgroupData<G>| {
                    let group_clone = master.group.clone();
                    DeserializeSeedSeq::new((0..).map(move |_| DeserializeWithGroup::new(group_clone.clone())), Vec::new(), |mut current, next| { current.push(next); current })
                },
                group: G: |master: &DeserializeSeedSubgroupData<G>| {
                    let group_clone = master.group.clone();
                    DeserializeSeedMapped::new(PhantomData::<()>, move |()| group_clone)
                }
            } where G: AbelianGroupStore + Clone, G::Group: SerializableElementGroup
        }

        DeserializeSeedNewtypeStruct::new(
            "Subgroup",
            DeserializeSeedDependentTuple::new(PhantomData::<G>, |group| DeserializeSeedSubgroupData { group }),
        )
        .deserialize(deserializer)
        .map(|data| Subgroup::new(data.group, data.order_multiple, data.generators).into())
    }
}

impl<G: AbelianGroupStore> Clone for SubgroupBase<G>
where
    G: Clone,
{
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            generators: self.generators.iter().map(|g| g.clone()).collect(),
            order_factorization: self.order_factorization.clone(),
            order_multiple: self.order_multiple.clone(),
            padic_generating_sets: self
                .padic_generating_sets
                .iter()
                .map(|sets| sets.iter().map(|set| set.iter().map(|g| g.clone()).collect()).collect())
                .collect(),
            padic_relation_lattices: self
                .padic_relation_lattices
                .iter()
                .map(|x| x.iter().map(|x| x.clone()).collect())
                .collect(),
            global_relation_lattice: self.global_relation_lattice.clone(),
            cyclic_decomposition: self.cyclic_decomposition.clone(),
            subgroup_order: self.subgroup_order.clone(),
        }
    }
}

impl<R> Subgroup<MultGroup<R>>
where
    R: RingStore,
    R::Ring: ZnRing + HashableElRing + DivisibilityRing,
{
    /// Creates a [`Subgroup`] of the given multiplicative group of a given ring `Z/nZ`.
    ///
    /// This will factor the modulus `n` and (p - 1) for each p|n, which may be expensive.
    #[stability::unstable(feature = "enable")]
    pub fn for_zn(group: MultGroup<R>, generators: Vec<GEl<MultGroup<R>>>) -> Self {
        let mut result = Self::for_zn_with_factorization(group).0;
        let order = result.order_multiple().clone();
        for g in generators {
            result = result.add_generator(g, &order);
        }
        return result;
    }

    fn for_zn_with_factorization(
        group: MultGroup<R>,
    ) -> (Self, Vec<(El<BigIntRing>, usize)>, Vec<Vec<(El<BigIntRing>, usize)>>) {
        let n_factorization = factor(ZZbig, group.underlying_ring().size(ZZbig).unwrap());
        let order_factorizations = n_factorization
            .iter()
            .map(|(p, e)| {
                let mut factorization = factor(ZZbig, ZZbig.sub_ref_fst(p, ZZbig.one()))
                    .into_iter()
                    .chain([(p.clone(), e - 1)].into_iter())
                    .filter(|(_, e)| *e > 0)
                    .collect::<Vec<_>>();
                factorization.sort_unstable_by(|(l, _), (r, _)| ZZbig.cmp(l, r));
                return factorization;
            })
            .collect::<Vec<_>>();
        let max_order_factorization = order_factorizations
            .iter()
            .fold(Vec::new(), |lhs: Vec<_>, rhs: &Vec<_>| {
                let mut result = Vec::new();
                let mut lhs_it = lhs.into_iter().peekable();
                let mut rhs_it = rhs.iter().cloned().peekable();
                while let Some((p_l, e_l)) = lhs_it.peek()
                    && let Some((p_r, e_r)) = rhs_it.peek()
                {
                    match ZZbig.cmp(p_l, p_r) {
                        Ordering::Less => {
                            result.push(lhs_it.next().unwrap());
                        }
                        Ordering::Greater => {
                            result.push(rhs_it.next().unwrap());
                        }
                        Ordering::Equal => {
                            let e = usize::max(*e_l, *e_r);
                            let p = lhs_it.next().unwrap().0;
                            _ = rhs_it.next().unwrap();
                            result.push((p, e));
                        }
                    }
                }
                result.extend(lhs_it);
                result.extend(rhs_it);
                return result;
            });
        return (
            Self::from(SubgroupBase {
                parent: group,
                generators: Vec::new(),
                order_multiple: ZZbig.prod(max_order_factorization.iter().map(|(p, e)| ZZbig.pow(p.clone(), *e))),
                order_factorization: max_order_factorization,
                padic_generating_sets: Vec::new(),
                padic_relation_lattices: Vec::new(),
                global_relation_lattice: OnceLock::new(),
                cyclic_decomposition: OnceLock::new(),
                subgroup_order: OnceLock::new(),
            }),
            n_factorization,
            order_factorizations,
        );
    }

    /// Creates a [`Subgroup`] that is the full unit group of the given ring `Z/nZ`.
    ///
    /// This will factor the modulus `n` and (p - 1) for each p|n, which may be expensive.
    #[stability::unstable(feature = "enable")]
    pub fn zn_unit_group(group: MultGroup<R>) -> Self {
        let (mut result, n_factorization, order_factorizations) = Self::for_zn_with_factorization(group);
        let order_multiple = result.order_multiple().clone();
        let two = ZZbig.int_hom().map(2);
        for ((p, e), order_factorization) in n_factorization.into_iter().zip(order_factorizations) {
            let ZZ = result.parent().underlying_ring().integer_ring();
            let n = int_cast(result.parent().underlying_ring().modulus().clone(), ZZbig, ZZ);
            let pe = ZZbig.pow(p.clone(), e);
            let rest = ZZbig.checked_div(&n, &pe).unwrap();
            if ZZbig.eq_el(&p, &two) {
                let Zn = result.parent().underlying_ring();
                let g1 = ZZbig.inv_crt([&ZZbig.int_hom().map(5), &ZZbig.one()], [&pe, &rest]);
                let g1 = result
                    .parent()
                    .from_ring_el(Zn.coerce(Zn.integer_ring(), int_cast(g1, Zn.integer_ring(), ZZbig)))
                    .unwrap();
                let g2 = result.parent().from_ring_el(Zn.neg_one()).unwrap();
                result = result.add_generator(g1, &order_multiple);
                result = result.add_generator(g2, &order_multiple);
            } else {
                let g = generator_mod_pe(p, e, &order_factorization);
                let g = ZZbig.inv_crt([&g, &ZZbig.one()], [&pe, &rest]);
                let Zn = result.parent().underlying_ring();
                let g = result
                    .parent()
                    .from_ring_el(Zn.coerce(Zn.integer_ring(), int_cast(g, Zn.integer_ring(), ZZbig)))
                    .unwrap();
                result = result.add_generator(g, &order_multiple);
            }
        }
        return result;
    }
}

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
/// # use feanor_math::group::*;
/// # use feanor_math::prelude::*;
/// let group = AddGroup::new(ZZi64);
/// assert_eq!(
///     Some(vec![1]),
///     baby_giant_step(&group, 0, &[0], &[ZZbig.power_of_two(10)])
/// );
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
/// ```rust
/// # use feanor_math::prelude::*;
/// # use feanor_math::group::*;
/// # use feanor_math::ring_impls::zn::*;
/// # use feanor_math::ring_impls::zn::zn_64b::*;
/// # use feanor_math::wrapper::*;
/// # use feanor_math::algorithms::discrete_log::*;
/// let ring = Zn64B::new(17);
/// let group = MultGroup::new(ring);
/// let x = group.from_ring_el(ring.int_hom().map(9)).unwrap();
/// assert_eq!(
///     Some(vec![3]),
///     baby_giant_step(
///         &group,
///         group.pow(&x, &int_cast(3, ZZbig, ZZi64)),
///         &[x],
///         &[ZZbig.power_of_two(4)]
///     )
/// );
/// ```
#[stability::unstable(feature = "enable")]
pub fn baby_giant_step<G>(group: G, value: GEl<G>, generators: &[GEl<G>], dlog_bounds: &[i64]) -> Option<Vec<i64>>
where
    G: AbelianGroupStore,
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
    let ns = dlog_bounds
        .iter()
        .map(|n| root_floor(ZZi64, *n, 2) + 1)
        .collect::<Vec<_>>();
    let count = int_cast(ZZbig.prod(ns.iter().map(|n| int_cast(*n, ZZbig, ZZi64))), ZZi64, ZZbig);
    let mut baby_step_table: HashMap<HashableGroupEl<_>, i64> = HashMap::with_capacity(count as usize);

    // fill baby step table
    {
        let mut current_els = (0..n).map(|_| value.clone()).collect::<Vec<_>>();
        let mut current_idxs = (0..n).map(|_| 0).collect::<Vec<_>>();
        for idx in 0..count {
            _ = baby_step_table.insert(HashableGroupEl::new(&group, current_els[n - 1].clone()), idx);

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
                current_els[j] = current_els[i].clone();
            }
        }
    }

    let giant_steps = generators
        .iter()
        .zip(ns.iter())
        .map(|(g, n)| group.pow(g.clone(), *n))
        .collect::<Vec<_>>();
    // iterate through giant steps
    {
        let start_el = giant_steps.iter().fold(group.identity(), |x, y| group.op_ref_snd(x, y));
        let mut current_els = (0..n).map(|_| start_el.clone()).collect::<Vec<_>>();
        let mut current_idxs = (0..n).map(|_| 1).collect::<Vec<_>>();
        for idx in 0..count {
            if let Some(bs_idx) = baby_step_table.get(&HashableGroupEl::new(&group, current_els[n - 1].clone())) {
                let mut bs_idx = *bs_idx;
                let mut result = current_idxs.clone();
                for j in (0..n).rev() {
                    let bs_idxs_j = bs_idx % ns[j];
                    bs_idx = bs_idx / ns[j];
                    result[j] = result[j] * ns[j] - bs_idxs_j;
                }
                if (0..dlog_bounds.len()).all(|j| result[j] <= dlog_bounds[j]) {
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
                current_els[j] = current_els[i].clone();
            }
        }
    }

    return None;
}

fn generator_mod_pe(p: El<BigIntRing>, e: usize, order_factorization: &[(El<BigIntRing>, usize)]) -> El<BigIntRing> {
    assert!(!ZZbig.eq_el(&p, &ZZbig.int_hom().map(2)));
    let mut rng = Rand64::new(0);
    let p_e_neg_1 = ZZbig.pow(p.clone(), e - 1);
    let pe = ZZbig.mul_ref(&p, &p_e_neg_1);
    let order = ZZbig.sub_ref_fst(&pe, p_e_neg_1);
    let Zpe = ZnGB::new(ZZbig, pe.clone());
    let mut generator = Zpe.random_element(|| rng.rand_u64());
    let powers = order_factorization
        .iter()
        .map(|(p_, _)| ZZbig.checked_div(&order, &p_).unwrap())
        .collect::<Vec<_>>();
    for _ in 0..PROBABILISTIC_REPETITIONS {
        if Zpe.is_one(&Zpe.pow_bigint(generator.clone(), &order))
            && powers
                .iter()
                .all(|power| !Zpe.is_one(&Zpe.pow_bigint(generator.clone(), power)))
        {
            return Zpe.smallest_positive_lift(generator);
        } else {
            generator = Zpe.random_element(|| rng.rand_u64());
        }
    }
    unreachable!()
}

#[cfg(test)]
struct ProdGroupBase<G: AbelianGroupStore, const N: usize>(G);

#[cfg(test)]
impl<G: AbelianGroupStore, const N: usize> PartialEq for ProdGroupBase<G, N> {
    fn eq(&self, other: &Self) -> bool { self.0.get_group() == other.0.get_group() }
}

#[cfg(test)]
impl<G: AbelianGroupStore, const N: usize> Debug for ProdGroupBase<G, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "({:?})^{}", self.0.get_group(), N) }
}

#[cfg(test)]
impl<G: AbelianGroupStore, const N: usize> AbelianGroupBase for ProdGroupBase<G, N> {
    type Element = [GEl<G>; N];

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool { (0..N).all(|i| self.0.eq_el(&lhs[i], &rhs[i])) }

    fn op(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        from_fn(|i| self.0.op_ref(&lhs[i], &rhs[i]))
    }

    fn hash<H: std::hash::Hasher>(&self, x: &Self::Element, hasher: &mut H) {
        for i in 0..N {
            self.0.hash(&x[i], hasher)
        }
    }

    fn inv(&self, x: &Self::Element) -> Self::Element { from_fn(|i| self.0.inv(&x[i])) }

    fn identity(&self) -> Self::Element { from_fn(|_| self.0.identity()) }

    fn fmt_el<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        let mut seq = out.debug_list();
        for x in value {
            _ = seq.entry(&self.0.formatted_el(x));
        }
        return seq.finish();
    }
}

#[cfg(test)]
use std::array::from_fn;

#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;
#[cfg(test)]
use crate::algorithms::linsolve::LinSolveRingStore;
#[cfg(test)]
use crate::algorithms::matmul::ComputeInnerProduct;
#[cfg(test)]
use crate::group::AddGroup;
#[cfg(test)]
use crate::ring_impls::zn::zn_static::Zn;

#[test]
fn test_baby_giant_step() {
    feanor_tracing::DelayedLogger::init_test();
    for base_bound in [21, 26, 31, 37] {
        let G = AddGroup::new(ZZi64);
        assert_eq!(Some(vec![6]), baby_giant_step(&G, 6, &[1], &[base_bound]));
        assert_eq!(None, baby_giant_step(&G, 0, &[1], &[base_bound]));

        let G = AddGroup::new(Zn::<20>::RING);
        assert_eq!(Some(vec![20]), baby_giant_step(&G, 0, &[1], &[base_bound]));
        assert_eq!(Some(vec![10]), baby_giant_step(&G, 10, &[1], &[base_bound]));
        assert_eq!(Some(vec![5]), baby_giant_step(&G, 0, &[16], &[base_bound]));
    }

    let G = AddGroup::new(ZZi64);

    // the collision point is at 96
    assert_eq!(Some(vec![9 - 1, 6 - 1]), baby_giant_step(&G, 85, &[10, 1], &[8, 8]));
    // the collision point is at 105
    assert_eq!(Some(vec![10 - 2, 5 - 0]), baby_giant_step(&G, 85, &[10, 1], &[21, 21]));
    // the collision point is at 90
    assert_eq!(Some(vec![6 - 0, 30 - 5]), baby_giant_step(&G, 85, &[10, 1], &[31, 31]));
}

#[test]
fn test_padic_relation_lattice() {
    feanor_tracing::DelayedLogger::init_test();
    let G = AddGroup::new(Zn::<81>::RING);
    let inner_prod = |l: Column<_, El<BigIntRing>>, r: &[i64]| {
        ZZi64.get_ring().inner_product(
            l.as_iter()
                .map(|x| int_cast(x.clone(), ZZi64, ZZbig))
                .zip(r.iter().copied()),
        )
    };

    let subgroup = Subgroup::new(&G, int_cast(81, ZZbig, ZZi64), vec![1]);
    assert_el_eq!(
        ZZbig,
        ZZbig.int_hom().map(-81),
        subgroup.get_group().padic_relation_lattices[0][4].at(0, 0)
    );
    assert_el_eq!(
        ZZbig,
        ZZbig.int_hom().map(-27),
        subgroup.get_group().padic_relation_lattices[0][3].at(0, 0)
    );
    assert_el_eq!(
        ZZbig,
        ZZbig.int_hom().map(-9),
        subgroup.get_group().padic_relation_lattices[0][2].at(0, 0)
    );
    assert_el_eq!(
        ZZbig,
        ZZbig.int_hom().map(-3),
        subgroup.get_group().padic_relation_lattices[0][1].at(0, 0)
    );
    assert_el_eq!(
        ZZbig,
        ZZbig.int_hom().map(-1),
        subgroup.get_group().padic_relation_lattices[0][0].at(0, 0)
    );

    let subgroup = Subgroup::new(&G, int_cast(81, ZZbig, ZZi64), vec![3, 6]);
    let matrix = &subgroup.get_group().padic_relation_lattices[0][4];
    let expected = OwnedMatrix::new(vec![1, 0, 13, 27], 2, 2).map(|x| int_cast(x, ZZbig, ZZi64));
    assert!(lattice_eq(ZZbig, expected.data(), matrix.data(), None, None));
    assert_eq!(0, inner_prod(matrix.data().col_at(1), &[3, 6]) % 81);

    let subgroup = Subgroup::new(&G, int_cast(81, ZZbig, ZZi64), vec![3, 9]);
    let matrix = &subgroup.get_group().padic_relation_lattices[0][4];
    let expected = OwnedMatrix::new(vec![3, 0, -1, 9], 2, 2).map(|x| int_cast(x, ZZbig, ZZi64));
    assert!(lattice_eq(ZZbig, expected.data(), matrix.data(), None, None));
    assert_eq!(0, inner_prod(matrix.data().col_at(1), &[3, 9]) % 81);

    let subgroup = Subgroup::new(&G, int_cast(81, ZZbig, ZZi64), vec![6, 18, 9]);
    let matrix = &subgroup.get_group().padic_relation_lattices[0][4];
    let expected = OwnedMatrix::new(vec![0, 3, 0, 1, -1, 0, -2, 0, 9], 3, 3).map(|x| int_cast(x, ZZbig, ZZi64));
    assert!(lattice_eq(ZZbig, expected.data(), matrix.data(), None, None));
    assert_eq!(0, inner_prod(matrix.data().col_at(1), &[6, 18, 9]) % 81);
    assert_eq!(0, inner_prod(matrix.data().col_at(2), &[6, 18, 9]) % 81);

    let G = GroupValue::from(ProdGroupBase(AddGroup::new(Zn::<81>::RING)));

    let subgroup = Subgroup::new(&G, int_cast(81, ZZbig, ZZi64), vec![[1, 4], [1, 1]]);
    let matrix = &subgroup.get_group().padic_relation_lattices[0][4];
    let expected = OwnedMatrix::new(vec![81, 27, 0, -27], 2, 2).map(|x| int_cast(x, ZZbig, ZZi64));
    assert!(lattice_eq(ZZbig, expected.data(), matrix.data(), None, None));
    assert_eq!(0, inner_prod(matrix.data().col_at(1), &[1, 1]) % 81);
    assert_eq!(0, inner_prod(matrix.data().col_at(1), &[4, 1]) % 81);

    let G = GroupValue::from(ProdGroupBase(AddGroup::new(Zn::<8>::RING)));

    let subgroup = Subgroup::new(&G, int_cast(8, ZZbig, ZZi64), vec![[6, 3, 5], [6, 2, 6], [4, 5, 7]]);
    let matrix = &subgroup.get_group().padic_relation_lattices[0][3];
    assert_eq!(0, inner_prod(matrix.data().col_at(1), &[6, 6, 4]) % 8);
    assert_eq!(0, inner_prod(matrix.data().col_at(1), &[3, 2, 5]) % 8);
    assert_eq!(0, inner_prod(matrix.data().col_at(1), &[5, 6, 7]) % 8);
    assert_eq!(0, inner_prod(matrix.data().col_at(2), &[6, 6, 4]) % 8);
    assert_eq!(0, inner_prod(matrix.data().col_at(2), &[3, 2, 5]) % 8);
    assert_eq!(0, inner_prod(matrix.data().col_at(2), &[5, 6, 7]) % 8);
}

#[test]
fn test_dlog() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = Zn::<153>::RING;
    let group = AddGroup::new(ring);
    let subgroup = Subgroup::new(&group, int_cast(3 * 17, ZZbig, ZZi64), vec![3]);
    assert!(subgroup.dlog(&1).is_none());
    assert!(subgroup.dlog(&17).is_none());
    assert_gel_eq!(group, 3, group.pow_bigint(3, &subgroup.dlog(&3).unwrap()[0]));
    let subgroup = subgroup.add_generator(17, &int_cast(9, ZZbig, ZZi64));
    assert_gel_eq!(
        group,
        17,
        group.op(
            group.pow_bigint(3, &subgroup.dlog(&17).unwrap()[0]),
            group.pow_bigint(17, &subgroup.dlog(&17).unwrap()[1])
        )
    );
}

#[test]
fn random_test_dlog() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = Zn::<1400>::RING;
    let int_hom = ring.can_hom(&ZZbig).unwrap();
    let mut rng = Rand64::new(0);
    let G = GroupValue::from(ProdGroupBase(AddGroup::new(ring)));
    let rand_gs = |rng: &mut Rand64| from_fn::<_, 3, _>(|_| ring.random_element(|| rng.rand_u64()));

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let gs = from_fn::<_, 3, _>(|_| rand_gs(&mut rng));
        let subgroup = Subgroup::new(&G, int_cast(1400, ZZbig, ZZi64), gs.into());

        let coeffs = rand_gs(&mut rng);
        let val = (0..3).fold(G.identity(), |current, i| {
            G.op(current, G.pow(gs[i].clone(), coeffs[i] as i64))
        });
        let dlog = subgroup.dlog(&val);
        println!(
            "{:?} * x + {:?} * y + {:?} * z = {:?} mod 1400",
            gs[0], gs[1], gs[2], val
        );
        if let Some(dlog) = dlog {
            for k in 0..3 {
                assert_el_eq!(
                    ring,
                    val[k],
                    ring.sum([
                        int_hom.mul_ref_map(&gs[0][k], &dlog[0]),
                        int_hom.mul_ref_map(&gs[1][k], &dlog[1]),
                        int_hom.mul_ref_map(&gs[2][k], &dlog[2])
                    ])
                );
            }
            println!("checked solution");
        }
    }

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let gs = from_fn::<_, 3, _>(|_| rand_gs(&mut rng));
        let subgroup = Subgroup::new(&G, int_cast(1400, ZZbig, ZZi64), gs.into());

        let val = rand_gs(&mut rng);
        let dlog = subgroup.dlog(&val);
        println!(
            "{:?} * x + {:?} * y + {:?} * z = {:?} mod 1400",
            gs[0], gs[1], gs[2], val
        );
        if let Some(dlog) = dlog {
            for k in 0..3 {
                assert_el_eq!(
                    ring,
                    val[k],
                    ring.sum([
                        int_hom.mul_ref_map(&gs[0][k], &dlog[0]),
                        int_hom.mul_ref_map(&gs[1][k], &dlog[1]),
                        int_hom.mul_ref_map(&gs[2][k], &dlog[2])
                    ])
                );
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
                    assert_el_eq!(
                        ring,
                        val[k],
                        ring.sum([
                            ring.mul(gs[0][k], *res.at(0, 0)),
                            ring.mul(gs[1][k], *res.at(1, 0)),
                            ring.mul(gs[2][k], *res.at(2, 0))
                        ])
                    );
                }
                assert!(solved == crate::algorithms::linsolve::SolveResult::NoSolution);
            }
            println!("has no solution");
        }
    }
}

#[test]
fn test_full_subgroup() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = Zn::<153>::RING;
    let group = MultGroup::new(ring);
    assert_el_eq!(
        ZZbig,
        ZZbig.int_hom().map(96),
        Subgroup::zn_unit_group(group).subgroup_order()
    );

    let ring = Zn::<1400>::RING;
    let group = MultGroup::new(ring);
    assert_el_eq!(
        ZZbig,
        ZZbig.int_hom().map(16 * 5 * 6),
        Subgroup::zn_unit_group(group).subgroup_order()
    );

    let ring = Zn::<257>::RING;
    let group = MultGroup::new(ring);
    assert_el_eq!(
        ZZbig,
        ZZbig.int_hom().map(256),
        Subgroup::zn_unit_group(group).subgroup_order()
    );
}

#[test]
fn test_subgroup_order() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = Zn::<153>::RING;
    let group = MultGroup::new(ring);
    let g1 = group.from_ring_el(ring.int_hom().map(2)).unwrap();
    let g2 = group.from_ring_el(ring.int_hom().map(37)).unwrap();

    let mut subgroup = Subgroup::for_zn(group.clone(), Vec::new());
    let order_multiple = subgroup.order_multiple().clone();
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(1), subgroup.subgroup_order());

    let next_gen = g1.clone();
    subgroup = subgroup.add_generator(next_gen, &order_multiple);
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(24), subgroup.subgroup_order());

    let next_gen = subgroup.parent().pow(g1.clone(), 2);
    subgroup = subgroup.add_generator(next_gen, &order_multiple);
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(24), subgroup.subgroup_order());

    let next_gen = g2.clone();
    subgroup = subgroup.add_generator(next_gen, &order_multiple);
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(96), subgroup.subgroup_order());

    let generating_set = Subgroup::for_zn(group, vec![g2]);
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(16), generating_set.subgroup_order());
}

#[test]
fn test_global_relation_lattice() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = Zn::<153>::RING;
    let group = MultGroup::new(ring);
    let g1 = group.from_ring_el(ring.int_hom().map(2)).unwrap();
    let g2 = group.from_ring_el(ring.int_hom().map(37)).unwrap();

    let subgroup = Subgroup::for_zn(group, vec![g1.clone(), g2.clone()]);
    let actual = subgroup.get_group().relation_lattice();
    let expected = OwnedMatrix::new(vec![6, 0, -4, 16], 2, 2).map(|x| int_cast(x, ZZbig, ZZi64));
    assert!(lattice_eq(ZZbig, expected.data(), actual.data(), None, None));

    let g3 = group.from_ring_el(ring.int_hom().map(10)).unwrap();
    let subgroup = subgroup.add_generator(g3, &int_cast(16, ZZbig, ZZi64));
    let actual = subgroup.get_group().relation_lattice();
    let expected = OwnedMatrix::new(vec![6, 0, 30, -4, 16, -55, 0, 0, 1], 3, 3).map(|x| int_cast(x, ZZbig, ZZi64));
    assert!(lattice_eq(ZZbig, expected.data(), actual.data(), None, None));
}

#[test]
fn test_intersection() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = Zn::<7>::RING;
    let group = MultGroup::new(ring);
    let g1 = group.from_ring_el(ring.int_hom().map(2)).unwrap();
    let g2 = group.from_ring_el(ring.int_hom().map(5)).unwrap();
    let g3 = group.from_ring_el(ring.int_hom().map(2)).unwrap();
    let actual = Subgroup::for_zn(group, vec![g1]).intersection(&Subgroup::for_zn(group, vec![g2]));
    let expected = Subgroup::for_zn(group, vec![g3]);
    assert_eq!(expected.get_group(), actual.get_group());
    let mut elements = actual
        .enumerate_elements()
        .map(|x| ring.smallest_positive_lift(*group.as_ring_el(&x)))
        .collect::<Vec<_>>();
    elements.sort_unstable();
    assert_eq!(vec![1, 2, 4], elements);

    let ring = Zn::<153>::RING;
    let group = MultGroup::new(ring);
    let g1 = group.from_ring_el(ring.int_hom().map(2)).unwrap();
    let g2 = group.from_ring_el(ring.int_hom().map(5)).unwrap();
    let g3 = group.from_ring_el(ring.int_hom().map(13)).unwrap();
    let actual = Subgroup::for_zn(group, vec![g1]).intersection(&Subgroup::for_zn(group, vec![g2]));
    let expected = Subgroup::for_zn(group, vec![g3]);
    assert_eq!(expected.get_group(), actual.get_group());
}

#[test]
fn test_enumerate_elements() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = Zn::<45>::RING;
    let group = AddGroup::new(ring);

    let subgroup = Subgroup::new(group.clone(), int_cast(45, ZZbig, ZZi64), Vec::new());
    let elements = subgroup.enumerate_elements().collect::<Vec<_>>();
    assert_eq!(vec![ring.zero()], elements);

    let subgroup = Subgroup::new(group, int_cast(45, ZZbig, ZZi64), vec![9, 15]);
    let mut elements = subgroup.enumerate_elements().collect::<Vec<_>>();
    elements.sort_unstable();
    assert_eq!((0..45).step_by(3).collect::<Vec<_>>(), elements);

    let ring = Zn::<6>::RING;
    let group = AddGroup::new(ring);
    let subgroup = Subgroup::new(group, int_cast(6, ZZbig, ZZi64), vec![2, 2]);
    let mut elements = subgroup.enumerate_elements().collect::<Vec<_>>();
    elements.sort_unstable();
    assert_eq!(vec![0, 2, 4], elements);
}

#[test]
fn test_quotient_cyclic_decomposition() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = Zn::<45>::RING;
    let group = AddGroup::new(ring);

    let G = Subgroup::new(group, int_cast(45, ZZbig, ZZi64), vec![6]);
    let H = Subgroup::new(group, int_cast(45, ZZbig, ZZi64), vec![15]);
    let [(g, n)] = G.quotient_cyclic_decomposition(&H).try_into().unwrap();
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(5), n);
    assert!(g % 3 == 0);
    assert!(g % 5 != 0);

    let ring = Zn::<153>::RING;
    let group = MultGroup::new(ring);
    let g1 = group.from_ring_el(ring.int_hom().map(2)).unwrap();
    let g2 = group.from_ring_el(ring.int_hom().map(37)).unwrap();
    let G = Subgroup::for_zn(group, vec![g1, g2]);
    let H = Subgroup::for_zn(group, vec![group.pow(g1, 2), group.pow(group.op(g2, g1), 2)]);
    let [(h1, n1), (h2, n2)] = G.quotient_cyclic_decomposition(&H).try_into().unwrap();
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(2), n1);
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(2), n2);
    assert!(!group.is_identity(&h1));
    assert!(!group.is_identity(&h2));
    let elements = [group.identity(), h1, h2, group.op(h1, h2)];
    for a in &elements {
        for b in &elements {
            assert!(group.eq_el(a, b) || !H.contains(&group.op_ref(a, &group.inv(b))));
        }
    }

    let ring = Zn::<91>::RING;
    let group = MultGroup::new(ring);
    let g1 = group.from_ring_el(ring.int_hom().map(66)).unwrap();
    let g2 = group.from_ring_el(ring.int_hom().map(50)).unwrap();
    let g3 = group.from_ring_el(ring.int_hom().map(2)).unwrap();
    let G = Subgroup::for_zn(group, vec![g1, g2]);
    let H = Subgroup::for_zn(group, vec![g3]);
    let [(h1, n1), (h2, n2)] = G.quotient_cyclic_decomposition(&H).try_into().unwrap();
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(2), n1);
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(3), n2);
    assert!(!group.is_identity(&h1));
    assert!(!group.is_identity(&h2));
    let elements = [
        group.identity(),
        h1,
        h2,
        group.op(h1, h2),
        group.op(h2, h2),
        group.op(h1, group.op(h2, h2)),
    ];
    for a in &elements {
        for b in &elements {
            assert!(group.eq_el(a, b) || !H.contains(&group.op_ref(a, &group.inv(b))));
        }
    }

    let ring = Zn::<1729>::RING;
    let group = MultGroup::new(ring);
    let g1 = group.from_ring_el(ring.int_hom().map(248)).unwrap();
    let g2 = group.from_ring_el(ring.int_hom().map(1597)).unwrap();
    let g3 = group.from_ring_el(ring.int_hom().map(457)).unwrap();
    let G = Subgroup::for_zn(group, vec![g1, g2]);
    let H = Subgroup::for_zn(group, vec![g3]);
    let [(h1, n1), (h2, n2)] = G.quotient_cyclic_decomposition(&H).try_into().unwrap();
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(2), n1);
    assert_el_eq!(ZZbig, ZZbig.int_hom().map(3), n2);
    assert!(!group.is_identity(&h1));
    assert!(!group.is_identity(&h2));
    let elements = [
        group.identity(),
        h1,
        h2,
        group.op(h1, h2),
        group.op(h2, h2),
        group.op(h1, group.op(h2, h2)),
    ];
    for a in &elements {
        for b in &elements {
            assert!(group.eq_el(a, b) || !H.contains(&group.op_ref(a, &group.inv(b))));
        }
    }
}
