use std::mem::swap;
use std::sync::atomic::AtomicBool;

use tracing::instrument;

use crate::PROBABILISTIC_REPETITIONS;
use crate::algorithms::hensel::{FactorsNotCoprimeError, HenselLift};
use crate::algorithms::interpolate::product_except_one;
use crate::algorithms::poly_factor::FactorPolyField;
use crate::algorithms::poly_gcd::gcd_lift::*;
use crate::algorithms::poly_gcd::power_decomposition_lift::*;
use crate::algorithms::poly_gcd::*;
use crate::algorithms::poly_root::poly_root;
use crate::algorithms::primelist::*;
use crate::algorithms::rational_reconstruction::balanced_rational_reconstruction;
use crate::homomorphism::{CanHomFrom, LambdaHom};
use crate::ring_impls::as_field::*;
use crate::ring_impls::extension::extension_impl::FreeAlgebraImpl;
use crate::ring_impls::extension::galois_field::*;
use crate::ring_impls::extension::{FreeAlgebra, FreeAlgebraStore};
use crate::ring_impls::poly::dense_poly::DensePolyRing;
use crate::ring_impls::rational::*;
use crate::ring_impls::zn::zn_64b::Zn64B;
use crate::ring_impls::zn::zn_big::ZnGB;
use crate::ring_impls::zn::*;
use crate::seq::{VectorFn, VectorView};

const HOPE_FOR_SQUAREFREE_ATTEMPTS: usize = 1;
const BEST_EFFORT_SQUAREFREE_CHECKS: usize = 3;

#[derive(Debug, Copy, Clone)]
enum QuotientAtError {
    Ramified,
    ReductionNotWellDefined,
}

type ResidueField = GaloisField<AsField<FreeAlgebraImpl<AsField<Zn64B>, Vec<El<AsField<Zn64B>>>>>>;
type ResidueFieldPolyRing = DensePolyRing<ResidueField>;
type ResidueRing = FreeAlgebraImpl<ZnGB<BigIntRing>, Vec<El<ZnGB<BigIntRing>>>>;
type ResidueRingPolyRing = DensePolyRing<ResidueRing>;

fn check_error<F, R>(f: F) -> Result<R, ()>
where
    F: FnOnce(&AtomicBool) -> R,
{
    let error = AtomicBool::new(false);
    let result = f(&error);
    if error.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(());
    } else {
        return Ok(result);
    }
}

fn QQ_to_Zpe_hom<'a, R, I, F>(
    QQ: R,
    Fp: F,
    error: &'a AtomicBool,
) -> impl use<'a, R, I, F> + Homomorphism<R::Ring, F::Ring>
where
    R: 'a + RingStore<Ring = RationalFieldBase<I>>,
    I: 'a + RingStore,
    I::Ring: IntegerRing,
    F: 'a + RingStore,
    F::Ring: ZnRing + CanHomFrom<I::Ring>,
{
    LambdaHom::new(QQ, Fp, move |QQ, Fp, x| {
        let mod_p = Fp.can_hom(QQ.base_ring()).unwrap();
        let x_num = mod_p.map_ref(QQ.get_ring().num(x));
        let x_den = mod_p.map_ref(QQ.get_ring().den(x));
        if !Fp.is_unit(&x_den) {
            error.store(true, std::sync::atomic::Ordering::Relaxed);
            Fp.zero()
        } else {
            Fp.checked_div(&x_num, &x_den).unwrap()
        }
    })
}

fn K_to_GR_hom<'a, K, I, R>(
    number_field: K,
    galois_ring: R,
    error: &'a AtomicBool,
) -> impl use<'a, K, I, R> + Homomorphism<K::Ring, R::Ring>
where
    K: 'a + RingStore,
    K::Ring: FreeAlgebra + Field,
    BaseRingStore<K>: RingStore<Ring = RationalFieldBase<I>>,
    I: 'a + RingStore,
    I::Ring: IntegerRing,
    R: 'a + RingStore,
    R::Ring: FreeAlgebra,
    BaseRingBase<R>: ZnRing + CanHomFrom<I::Ring>,
{
    let mod_p = QQ_to_Zpe_hom(number_field.base_ring().clone(), galois_ring.base_ring().clone(), error);
    galois_ring.into_lifted_hom(number_field, mod_p).ok().unwrap()
}

struct ReconstructNumberFieldEl<'a, K, I>
where
    K: RingStore,
    K::Ring: FreeAlgebra + Field,
    BaseRingStore<K>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    number_field: K,
    residue_rings: Vec<&'a ResidueRingPolyRing>,
    ZpeX: DensePolyRing<ZnGB<BigIntRing>>,
    unit_vectors: Vec<El<DensePolyRing<ZnGB<BigIntRing>>>>,
    complete_product: El<DensePolyRing<ZnGB<BigIntRing>>>,
}

impl<'a, K, I> ReconstructNumberFieldEl<'a, K, I>
where
    K: RingStore,
    K::Ring: FreeAlgebra + Field,
    BaseRingStore<K>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    #[instrument(skip_all, level = "trace")]
    fn new(number_field: K, residue_rings: Vec<&'a ResidueRingPolyRing>) -> Self {
        let QQ = number_field.base_ring();
        let ZpeX = DensePolyRing::new(residue_rings[0].base_ring().base_ring().clone(), "X");
        let Zpe = ZpeX.base_ring();
        assert!(
            residue_rings
                .iter()
                .all(|GRX| GRX.base_ring().base_ring().get_ring() == Zpe.get_ring())
        );

        let mut unit_vectors = (0..residue_rings.len()).map(|_| ZpeX.zero()).collect::<Vec<_>>();
        product_except_one(
            &ZpeX,
            residue_rings
                .as_fn()
                .map_fn(|GRX| GRX.base_ring().generating_poly(&ZpeX, Zpe.identity())),
            &mut unit_vectors,
        );
        let complete_product = ZpeX.mul_ref_fst(
            &unit_vectors[0],
            residue_rings[0].base_ring().generating_poly(&ZpeX, Zpe.identity()),
        );
        check_error(|error| {
            debug_assert_el_eq!(
                &ZpeX,
                &complete_product,
                &number_field.generating_poly(&ZpeX, QQ_to_Zpe_hom(QQ, Zpe, &error))
            )
        })
        .unwrap();
        for (i, GRX) in residue_rings.iter().enumerate() {
            let inv_normalization_factor = ZpeX.evaluate(
                unit_vectors.at(i),
                &GRX.base_ring().canonical_gen(),
                GRX.base_ring().inclusion(),
            );
            let normalization_factor = GRX.base_ring().invert(&inv_normalization_factor).unwrap();
            let lifted_normalization_factor = GRX.base_ring().poly_repr(&ZpeX, &normalization_factor, Zpe.identity());
            let unreduced_new_unit_vector = ZpeX.mul(
                std::mem::replace(&mut unit_vectors[i], ZpeX.zero()),
                lifted_normalization_factor,
            );
            unit_vectors[i] = ZpeX.div_rem_monic(unreduced_new_unit_vector, &complete_product).1;
        }
        return Self {
            number_field,
            residue_rings,
            ZpeX,
            unit_vectors,
            complete_product,
        };
    }

    #[instrument(skip_all, level = "trace")]
    fn reconstruct(&self, residue_values: &[El<ResidueRing>]) -> El<K> {
        assert_eq!(self.residue_rings.len(), residue_values.len());
        let Zpe = self.ZpeX.base_ring();
        let QQ = self.number_field.base_ring();
        let ZZ = QQ.base_ring();
        let combined = <_ as RingStore>::sum(
            &self.ZpeX,
            self.residue_rings.iter().enumerate().map(|(i, GRX)| {
                let unreduced_result = self.ZpeX.mul_ref_snd(
                    GRX.base_ring()
                        .poly_repr(&self.ZpeX, &residue_values[i], Zpe.identity()),
                    &self.unit_vectors[i],
                );
                self.ZpeX.div_rem_monic(unreduced_result, &self.complete_product).1
            }),
        );
        for (i, GRX) in self.residue_rings.iter().enumerate() {
            debug_assert_el_eq!(
                GRX.base_ring(),
                &residue_values[i],
                &self
                    .ZpeX
                    .evaluate(&combined, &GRX.base_ring().canonical_gen(), GRX.base_ring().inclusion())
            );
        }
        // now lift the polynomial modulo `p^e` to the rationals
        let result = self
            .number_field
            .from_canonical_basis((0..self.number_field.rank()).map(|i| {
                let (num, den) = balanced_rational_reconstruction(Zpe, self.ZpeX.coefficient_at(&combined, i).clone());
                return QQ.div(
                    &QQ.inclusion().map(int_cast(num, ZZ, Zpe.integer_ring())),
                    &QQ.inclusion().map(int_cast(den, ZZ, Zpe.integer_ring())),
                );
            }));
        return result;
    }
}

struct ResidueFieldsAtPrime {
    FqXs: Vec<ResidueFieldPolyRing>,
}

impl ResidueFieldsAtPrime {
    #[instrument(skip_all, level = "trace")]
    fn new<K, I>(number_field: K, Fp: AsField<Zn64B>) -> Result<Self, QuotientAtError>
    where
        K: RingStore,
        K::Ring: FreeAlgebra + Field,
        BaseRingStore<K>: RingStore<Ring = RationalFieldBase<I>>,
        I: RingStore,
        I::Ring: IntegerRing,
    {
        let QQ = number_field.base_ring();
        let FpX = DensePolyRing::new(Fp, "X");
        let Fp = FpX.base_ring();
        let generating_poly = check_error(|error| number_field.generating_poly(&FpX, QQ_to_Zpe_hom(QQ, Fp, &error)))
            .map_err(|_| QuotientAtError::ReductionNotWellDefined)?;

        let generating_poly_factorization = FactorPolyField::factor_poly(&FpX, &generating_poly).0;
        if generating_poly_factorization.iter().any(|(_, e)| *e > 1) {
            return Err(QuotientAtError::Ramified);
        }

        return Ok(ResidueFieldsAtPrime {
            FqXs: generating_poly_factorization
                .into_iter()
                .map(|(f, _)| {
                    let degree = FpX.degree(&f).unwrap();
                    let modulus = (0..degree)
                        .map(|i| Fp.negate(FpX.coefficient_at(&f, i).clone()))
                        .collect::<Vec<_>>();
                    let Fq = GaloisField::create(FreeAlgebraImpl::new(Fp.clone(), degree, modulus).as_field().unwrap());
                    let FqX = DensePolyRing::new(Fq, "X");
                    return FqX;
                })
                .collect(),
        });
    }
}

struct ResidueRingsAtPrimePower {
    lifter: HenselLift<DensePolyRing<ZnGB<BigIntRing>>>,
    prime: El<BigIntRing>,
}

impl ResidueRingsAtPrimePower {
    #[instrument(skip_all, level = "trace")]
    fn new(residue_fields: &ResidueFieldsAtPrime) -> Self {
        let Fp = residue_fields.FqXs[0].base_ring().base_ring().clone();
        let FpX = DensePolyRing::new(Fp, "X");
        let Zp = ZnGB::new(ZZbig, int_cast(*Fp.modulus(), ZZbig, ZZi64));
        let ZpX = DensePolyRing::new(Zp.clone(), "X");
        let gen_poly_factorization = residue_fields
            .FqXs
            .iter()
            .map(|FqX| FqX.base_ring().generating_poly(&FpX, FpX.base_ring().identity()))
            .collect::<Vec<_>>();
        let gen_poly_lifter = HenselLift::new(FpX, gen_poly_factorization).unwrap();
        return Self {
            lifter: gen_poly_lifter.change_ring(ZpX, ZnReductionMap::new(&Fp, &Zp).unwrap()),
            prime: Zp.modulus().clone(),
        };
    }

    #[instrument(skip_all, level = "trace")]
    fn lift_genpoly_factorization<K, I>(&mut self, number_field: K, lift_to_degree: usize)
    where
        K: RingStore,
        K::Ring: Field + FreeAlgebra,
        BaseRingStore<K>: RingStore<Ring = RationalFieldBase<I>>,
        I: RingStore,
        I::Ring: IntegerRing,
    {
        let QQ = number_field.base_ring();
        let Zpe = ZnGB::new(ZZbig, ZZbig.pow(self.prime.clone(), lift_to_degree));
        let ZpeX = DensePolyRing::new(Zpe.clone(), "X");

        let target_mod_pe =
            check_error(|error| number_field.generating_poly(&ZpeX, QQ_to_Zpe_hom(QQ, &Zpe, &error))).unwrap();

        take_mut::take(&mut self.lifter, |lifter| {
            lifter.lift_to(lift_to_degree, ZpeX, &target_mod_pe, |old_base_ring, Zpe, x| {
                Zpe.get_ring()
                    .from_int_promise_reduced(old_base_ring.smallest_positive_lift(x.clone()))
            })
        });
    }

    fn ZpeX(&self) -> &DensePolyRing<ZnGB<BigIntRing>> { self.lifter.poly_ring() }

    #[instrument(skip_all, level = "trace")]
    fn create_residue_rings(&self) -> Vec<ResidueRingPolyRing> {
        self.lifter
            .factorization()
            .map(|f| {
                let GR = FreeAlgebraImpl::new(
                    self.ZpeX().base_ring().clone(),
                    self.ZpeX().degree(f).unwrap(),
                    (0..self.ZpeX().degree(f).unwrap())
                        .map(|i| self.ZpeX().base_ring().negate(self.ZpeX().coefficient_at(f, i).clone()))
                        .collect(),
                );
                let GRX = DensePolyRing::new(GR, "X");
                return GRX;
            })
            .collect::<Vec<_>>()
    }
}

struct NumberFieldFactorizationLift {
    gen_poly_lift: ResidueRingsAtPrimePower,
    power_factor_lifters: Vec<HenselLift<ResidueRingPolyRing>>,
}

impl NumberFieldFactorizationLift {
    #[instrument(skip_all, level = "trace")]
    fn new(
        residue_fields: &ResidueFieldsAtPrime,
        factorizations: Vec<Vec<El<ResidueFieldPolyRing>>>,
    ) -> Result<Self, FactorsNotCoprimeError> {
        assert_eq!(residue_fields.FqXs.len(), factorizations.len());
        let mut lifters = Vec::new();
        for (factorization, FqX) in factorizations.into_iter().zip(&residue_fields.FqXs) {
            lifters.push(HenselLift::new(FqX, factorization)?);
        }
        let gen_poly_lift = ResidueRingsAtPrimePower::new(residue_fields);
        let residue_rings = gen_poly_lift.create_residue_rings();
        debug_assert_eq!(residue_fields.FqXs.len(), residue_rings.len());
        let lifters = lifters
            .into_iter()
            .zip(residue_fields.FqXs.iter().zip(residue_rings))
            .map(|(lifter, (FqX, GRX))| {
                let Fq = FqX.base_ring();
                let GR = GRX.base_ring().clone();
                lifter.change_ring(
                    GRX,
                    GR.lifted_hom(Fq, ZnReductionMap::new(Fq.base_ring(), GR.base_ring()).unwrap())
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        return Ok(Self {
            gen_poly_lift,
            power_factor_lifters: lifters,
        });
    }

    #[instrument(skip_all, level = "trace")]
    fn lift_main_factorization<P, I>(&mut self, KX: P, target: &El<P>, lift_to_degree: usize)
    where
        P: RingStore + Copy,
        P::Ring: PolyRing,
        BaseRingBase<P>: Field + FreeAlgebra,
        BaseRingStore<BaseRingStore<P>>: RingStore<Ring = RationalFieldBase<I>>,
        I: RingStore,
        I::Ring: IntegerRing,
    {
        let K = KX.base_ring();
        self.gen_poly_lift.lift_genpoly_factorization(K, lift_to_degree);
        let new_residue_rings = self.gen_poly_lift.create_residue_rings();
        for (lifter, GRX) in self.power_factor_lifters.iter_mut().zip(new_residue_rings) {
            let target_mod_pe = check_error(|error| {
                GRX.lifted_hom(KX, K_to_GR_hom(K, GRX.base_ring(), &error))
                    .map_ref(target)
            })
            .unwrap();
            let hom = GRX.base_ring().base_ring().clone().into_can_hom(ZZbig).unwrap();
            take_mut::take(lifter, |lifter| {
                lifter.lift_to(lift_to_degree, GRX, &target_mod_pe, |old_base_ring, GR, x| {
                    GR.from_canonical_basis(
                        old_base_ring
                            .wrt_canonical_basis(x)
                            .iter()
                            .map(|c| hom.map(old_base_ring.base_ring().smallest_lift(c))),
                    )
                })
            });
        }
    }

    fn reconstruct_main_factorization<P, I>(&self, KX: P) -> Vec<El<P>>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing,
        BaseRingBase<P>: Field + FreeAlgebra,
        BaseRingStore<BaseRingStore<P>>: RingStore<Ring = RationalFieldBase<I>>,
        I: RingStore,
        I::Ring: IntegerRing,
    {
        let residue_rings = self
            .power_factor_lifters
            .iter()
            .map(|lifter| lifter.poly_ring())
            .collect::<Vec<_>>();
        let reconstruct_number_field_el = ReconstructNumberFieldEl::new(KX.base_ring(), residue_rings);

        let factor_count = self.power_factor_lifters[0].factorization().count();
        let factor_degrees = self.power_factor_lifters[0]
            .factorization()
            .map(|f| self.power_factor_lifters[0].poly_ring().degree(f).unwrap())
            .collect::<Vec<_>>();
        return (0..factor_count)
            .map(|i| {
                KX.from_terms((0..=factor_degrees[i]).map(|j| {
                    (
                        reconstruct_number_field_el.reconstruct(
                            &self
                                .power_factor_lifters
                                .iter()
                                .map(|lifter| {
                                    lifter
                                        .poly_ring()
                                        .coefficient_at(lifter.factorization().nth(i).unwrap(), j)
                                        .clone()
                                })
                                .collect::<Vec<_>>(),
                        ),
                        j,
                    )
                }))
            })
            .collect::<Vec<_>>();
    }
}

struct PolyGCDFactorizationLift {
    factorization_lift: NumberFieldFactorizationLift,
    is_lhs: bool,
}

impl PolyGCDFactorizationLift {
    #[instrument(skip_all, level = "trace")]
    fn new(
        residue_fields: ResidueFieldsAtPrime,
        lhs: &[El<ResidueFieldPolyRing>],
        rhs: &[El<ResidueFieldPolyRing>],
        gcd: Vec<El<ResidueFieldPolyRing>>,
    ) -> Result<Self, NotLiftable> {
        let mut factorizations = Vec::new();
        let mut gcd_deg = None;
        for ((lhs, gcd), FqX) in lhs.into_iter().zip(gcd.iter().cloned()).zip(&residue_fields.FqXs) {
            if let Some(gcd_deg) = gcd_deg {
                if FqX.degree(&gcd).unwrap() != gcd_deg {
                    return Err(NotLiftable::BadPrime);
                }
            } else {
                gcd_deg = Some(FqX.degree(&gcd).unwrap());
            }
            factorizations.push(vec![FqX.checked_div(&lhs, &gcd).unwrap(), gcd]);
        }
        if let Ok(factorization_lift) = NumberFieldFactorizationLift::new(&residue_fields, factorizations) {
            return Ok(Self {
                is_lhs: true,
                factorization_lift,
            });
        }
        let mut factorizations = Vec::new();
        for ((rhs, gcd), FqX) in rhs.into_iter().zip(gcd.iter().cloned()).zip(&residue_fields.FqXs) {
            factorizations.push(vec![FqX.checked_div(&rhs, &gcd).unwrap(), gcd]);
        }
        return Ok(Self {
            is_lhs: true,
            factorization_lift: NumberFieldFactorizationLift::new(&residue_fields, factorizations)
                .map_err(|_| NotLiftable::NotSquarefree)?,
        });
    }

    #[instrument(skip_all, level = "trace")]
    fn lift_gcd_factorization<P, I>(
        &mut self,
        KX: P,
        lhs: &El<P>,
        rhs: &El<P>,
        lift_to_degree: usize,
    ) -> Result<El<P>, LiftUnsuccessful>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingBase<P>: Field + FreeAlgebra,
        BaseRingStore<BaseRingStore<P>>: RingStore<Ring = RationalFieldBase<I>>,
        I: RingStore,
        I::Ring: IntegerRing,
    {
        let target = if self.is_lhs { lhs } else { rhs };
        self.factorization_lift
            .lift_main_factorization(KX, target, lift_to_degree);
        let [target_over_gcd, gcd] = self
            .factorization_lift
            .reconstruct_main_factorization(KX)
            .try_into()
            .ok()
            .unwrap();
        if KX.eq_el(&KX.mul_ref(&target_over_gcd, &gcd), target) {
            if (self.is_lhs && KX.divides(rhs, &gcd)) || (!self.is_lhs && KX.divides(lhs, &gcd)) {
                return Ok(gcd);
            } else {
                return Err(LiftUnsuccessful);
            }
        } else {
            return Err(LiftUnsuccessful);
        }
    }
}

/// Tries to compute the gcd of monic polynomials `f, g in Z[X]`.
///
/// This will fail if `lhs/d, d` and `rhs/d, d` are both not pairwise coprime, where `d = gcd(lhs,
/// rhs)`. It can, in theory, also fail in other settings, but the probability is very low for
/// larger values of `attempts`.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_gcd_number_field_squarefree<P, I>(KX: P, lhs: &El<P>, rhs: &El<P>, attempts: usize) -> PolyGCDResult<El<P>>
where
    P: RingStore + Copy,
    P::Ring: PolyRing + DivisibilityRing,
    BaseRingBase<P>: Field + FreeAlgebra,
    BaseRingStore<BaseRingStore<P>>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    if KX.is_zero(lhs) {
        return PolyGCDResult::FoundGCD(rhs.clone());
    } else if KX.is_zero(rhs) {
        return PolyGCDResult::FoundGCD(lhs.clone());
    }
    let K = KX.base_ring();

    let gcds_modulo_p = large_prime_fields()
        .filter_map(|Fp| {
            let residue_fields = ResidueFieldsAtPrime::new(K, Fp).ok()?;
            let poly_mod_p = |poly| {
                residue_fields
                    .FqXs
                    .iter()
                    .map(|FqX| {
                        check_error(|error| {
                            FqX.lifted_hom(KX, K_to_GR_hom(K, FqX.base_ring(), &error))
                                .map_ref(poly)
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            };
            let lhs_mod_p = poly_mod_p(lhs).ok()?;
            let rhs_mod_p = poly_mod_p(rhs).ok()?;
            let gcd = lhs_mod_p
                .iter()
                .zip(rhs_mod_p.iter())
                .zip(&residue_fields.FqXs)
                .map(|((l, r), FqX)| PolyTFracGCDRing::gcd(FqX, l, r))
                .collect::<Vec<_>>();
            let gcd_deg = gcd
                .iter()
                .zip(&residue_fields.FqXs)
                .map(|(gcd, FqX)| FqX.degree(gcd).unwrap())
                .min()
                .unwrap();
            return Some((
                PolyGCDSignature::new(gcd_deg),
                (residue_fields, lhs_mod_p, rhs_mod_p, gcd),
            ));
        })
        .take(attempts);
    poly_gcd_from_quotients(
        gcds_modulo_p,
        |(residue_fields, lhs, rhs, gcd)| PolyGCDFactorizationLift::new(residue_fields, &lhs, &rhs, gcd),
        |gcd_factorization_lift, lift_to_degree| {
            gcd_factorization_lift.lift_gcd_factorization(KX, lhs, rhs, lift_to_degree)
        },
    )
}

/// Computes the gcd of monic polynomials `f, g in K[X]` over a number field `K`.
///
/// Use this when implementing [`PolyTFracGCDRing`] for number fields; Otherwise, compute power
/// decompositions through [`PolyTFracGCDRing::gcd()`].
///
/// [`PolyTFracGCDRing`]: crate::algorithms::poly_gcd::PolyTFracGCDRing
/// [`PolyTFracGCDRing::gcd()`]: crate::algorithms::poly_gcd::PolyTFracGCDRing::gcd()
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_gcd_number_field<P, I>(KX: P, lhs: &El<P>, rhs: &El<P>) -> El<P>
where
    P: RingStore + Copy,
    P::Ring: PolyRing + DivisibilityRing,
    BaseRingBase<P>: Field + FreeAlgebra,
    BaseRingStore<BaseRingStore<P>>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    let mut lhs = KX.normalize(lhs.clone()).0;
    let mut rhs = KX.normalize(rhs.clone()).0;
    match poly_gcd_number_field_squarefree(KX, &lhs, &rhs, HOPE_FOR_SQUAREFREE_ATTEMPTS) {
        PolyGCDResult::TrivialGCD => return KX.one(),
        PolyGCDResult::FoundGCD(res) => return res,
        _ => {}
    }

    if KX.degree(&lhs).unwrap() > KX.degree(&rhs).unwrap() {
        swap(&mut lhs, &mut rhs);
    }
    let lhs_power_decomposition = poly_power_decomposition_number_field(KX, &lhs);
    let mut result = KX.one();
    for (fi, i) in &lhs_power_decomposition {
        for _ in 0..*i {
            match poly_gcd_number_field_squarefree(KX, &fi, &rhs, PROBABILISTIC_REPETITIONS) {
                PolyGCDResult::TrivialGCD => break,
                PolyGCDResult::FoundGCD(gcd_i) => {
                    rhs = KX.checked_div(&rhs, &gcd_i).unwrap();
                    KX.mul_assign(&mut result, gcd_i);
                }
                _ => unreachable!(),
            }
        }
    }
    return result;
}

struct PolyPowerDecompositionLift {
    factorizaton_lift: NumberFieldFactorizationLift,
    exponents: Vec<usize>,
}

impl PolyPowerDecompositionLift {
    #[instrument(skip_all, level = "trace")]
    fn new(
        residue_fields: ResidueFieldsAtPrime,
        power_decompositions: Vec<Vec<(El<ResidueFieldPolyRing>, usize)>>,
    ) -> Result<Self, BadPrime> {
        let mut exponents = None;
        let mut factorizations = Vec::new();
        for (mut power_decomp, FqX) in power_decompositions.into_iter().zip(&residue_fields.FqXs) {
            power_decomp.sort_unstable_by_key(|(_, e)| *e);
            let expected_exponents = power_decomp.iter().map(|(_, e)| *e).collect::<Vec<_>>();
            if let Some(exponents) = &exponents {
                if &expected_exponents != exponents {
                    return Err(BadPrime);
                }
            } else {
                exponents = Some(expected_exponents);
            }
            factorizations.push(power_decomp.into_iter().map(|(f, i)| FqX.pow(f, i)).collect());
        }
        return Ok(Self {
            exponents: exponents.unwrap(),
            factorizaton_lift: NumberFieldFactorizationLift::new(&residue_fields, factorizations).unwrap(),
        });
    }

    #[instrument(skip_all, level = "trace")]
    fn lift_power_decomposition<P, I>(
        &mut self,
        KX: P,
        target: &El<P>,
        lift_to_degree: usize,
    ) -> Result<Vec<(El<P>, usize)>, LiftUnsuccessful>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing,
        BaseRingBase<P>: Field + FreeAlgebra,
        BaseRingStore<BaseRingStore<P>>: RingStore<Ring = RationalFieldBase<I>>,
        I: RingStore,
        I::Ring: IntegerRing,
    {
        self.factorizaton_lift
            .lift_main_factorization(KX, target, lift_to_degree);
        let reconstructed_factorization = self.factorizaton_lift.reconstruct_main_factorization(KX);
        assert_eq!(self.exponents.len(), reconstructed_factorization.len());
        if KX.eq_el(&KX.prod(reconstructed_factorization.iter().cloned()), target) {
            return reconstructed_factorization
                .into_iter()
                .zip(self.exponents.iter())
                .map(|(f, i)| poly_root(KX, &f, *i).map(|f| (f, *i)).ok_or(LiftUnsuccessful))
                .collect();
        } else {
            return Err(LiftUnsuccessful);
        }
    }
}

/// Computes the power decomposition of polynomials `f, g in K[X]` over a number field `K`.
///
/// Use this when implementing [`PolyTFracGCDRing`] for number fields; Otherwise, compute power
/// decompositions through [`PolyTFracGCDRing::poly_power_decomposition()`].
///
/// [`PolyTFracGCDRing`]: crate::algorithms::poly_gcd::PolyTFracGCDRing
/// [`PolyTFracGCDRing::poly_power_decomposition()`]: crate::algorithms::poly_gcd::PolyTFracGCDRing::poly_power_decomposition()
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_power_decomposition_number_field<P, I>(KX: P, poly: &El<P>) -> Vec<(El<P>, usize)>
where
    P: RingStore + Copy,
    P::Ring: PolyRing,
    BaseRingBase<P>: Field + FreeAlgebra,
    BaseRingStore<BaseRingStore<P>>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    assert!(!KX.is_zero(poly));
    let poly = KX.normalize(poly.clone()).0;
    let K = KX.base_ring();

    let power_decompositions_modulo_p = large_prime_fields()
        .filter_map(|Fp| {
            let residue_fields = ResidueFieldsAtPrime::new(K, Fp).ok()?;
            let poly_mod_p = residue_fields
                .FqXs
                .iter()
                .map(|FqX| {
                    check_error(|error| {
                        FqX.lifted_hom(KX, K_to_GR_hom(K, FqX.base_ring(), &error))
                            .map_ref(&poly)
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .ok()?;
            let power_decompositions = poly_mod_p
                .into_iter()
                .zip(&residue_fields.FqXs)
                .map(|(f, FqX)| PolyTFracGCDRing::power_decomposition(FqX, &f))
                .collect::<Vec<_>>();
            let smallest_signature = power_decompositions
                .iter()
                .zip(&residue_fields.FqXs)
                .map(|(decomp, FqX)| PolyPowerDecompositionSignature::from_decomposition(FqX, decomp))
                .min_by_key(|sig| sig.signature_sum())
                .unwrap();
            return Some((smallest_signature, (residue_fields, power_decompositions)));
        })
        .take(PROBABILISTIC_REPETITIONS);
    poly_power_decomposition_from_quotients(
        power_decompositions_modulo_p,
        |(residue_fields, power_decompositions)| PolyPowerDecompositionLift::new(residue_fields, power_decompositions),
        |lift, lift_to_degree| lift.lift_power_decomposition(&KX, &poly, lift_to_degree),
    )
    .unwrap()
}

#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;
#[cfg(test)]
use crate::ring_impls::extension::number_field::*;
#[cfg(test)]
use crate::wrapper::RingElementWrapper;

#[cfg(test)]
fn test_field() -> NumberField<AsField<FreeAlgebraImpl<RationalField<BigIntRing>, [El<RationalField<BigIntRing>>; 1]>>> {
    let QQ = RationalField::new(ZZbig);
    NumberField::from(NumberFieldBase::create(AsField::from(
        AsFieldBase::promise_is_field(FreeAlgebraImpl::new(QQ, 4, [QQ.one()])).unwrap(),
    )))
}

#[test]
fn test_poly_power_decomposition_number_field() {
    feanor_tracing::DelayedLogger::init_test();
    let field = test_field();
    let poly_ring = DensePolyRing::new(field, "X");
    let zeta = RingElementWrapper::new(
        &poly_ring,
        poly_ring.inclusion().map(poly_ring.base_ring().canonical_gen()),
    );
    let [f1, f2, f3, f4] = poly_ring.with_wrapped_indeterminate(|X| {
        [
            X - &zeta,
            X + &zeta,
            X.pow_ref(2) + zeta.pow_ref(1) * X + 100,
            &zeta * X.pow_ref(3) - X + 1,
        ]
    });
    let multiply_out =
        |list: &[(El<DensePolyRing<_>>, usize)]| poly_ring.prod(list.iter().map(|(g, k)| poly_ring.pow(g.clone(), *k)));
    let assert_eq = |expected: &[(El<DensePolyRing<_>>, usize)], actual: &[(El<DensePolyRing<_>>, usize)]| {
        assert!(expected.is_sorted_by_key(|(_, k)| *k));
        assert!(actual.is_sorted_by_key(|(_, k)| *k));
        assert_eq!(expected.len(), actual.len());
        for ((f_expected, k_expected), (f_actual, k_actual)) in expected.iter().zip(actual.iter()) {
            assert_el_eq!(&poly_ring, f_expected, f_actual);
            assert_eq!(k_expected, k_actual);
        }
    };

    let expected = [(f1.clone(), 1)];
    let actual = poly_power_decomposition_number_field(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, &actual);

    let expected = [(poly_ring.normalize(poly_ring.mul_ref(&f3, &f4)).0, 3)];
    let actual = poly_power_decomposition_number_field(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, &actual);

    let expected = [(f2.clone(), 2), (poly_ring.normalize(poly_ring.mul_ref(&f3, &f4)).0, 3)];
    let actual = poly_power_decomposition_number_field(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, &actual);

    let expected = [
        (poly_ring.mul_ref(&f1, &f2), 1),
        (poly_ring.normalize(f4.clone()).0, 2),
        (f3.clone(), 3),
    ];
    let actual = poly_power_decomposition_number_field(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, &actual);

    let expected = [
        (poly_ring.mul_ref(&f1, &f2), 2),
        (poly_ring.normalize(f4.clone()).0, 4),
        (f3.clone(), 6),
    ];
    let actual = poly_power_decomposition_number_field(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, &actual);
}

#[test]
fn random_test_poly_power_decomposition_number_field() {
    feanor_tracing::DelayedLogger::init_test();
    let field = test_field();
    let poly_ring = DensePolyRing::new(field, "X");
    let mut rng = oorandom::Rand64::new(1);
    let bound = ZZbig.int_hom().map(500);
    let mut random_poly_of_deg = |deg: usize| {
        poly_ring.from_terms((0..=deg).map(|i| {
            (
                ring.from_canonical_basis((0..ring.rank()).map(|_| {
                    QQ.inclusion()
                        .map(QQ.base_ring().get_uniformly_random(&bound, || rng.rand_u64()))
                })),
                i,
            )
        }))
    };

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let f = random_poly_of_deg(5);
        let g = random_poly_of_deg(3);
        let h = random_poly_of_deg(2);
        let poly = poly_ring.prod([&f, &g, &g, &h, &h, &h].into_iter().map(|poly| poly.clone()));
        let power_decomp = poly_power_decomposition_number_field(&poly_ring, &poly);

        assert_el_eq!(
            &poly_ring,
            &poly_ring.normalize(poly).0,
            poly_ring.prod(power_decomp.iter().map(|(poly, k)| poly_ring.pow(poly.clone(), *k)))
        );
        assert!(
            poly_ring.divides(
                &poly_ring.prod(
                    power_decomp
                        .iter()
                        .filter(|(_, k)| k % 3 == 0)
                        .map(|(poly, k)| poly_ring.pow(poly.clone(), k / 3))
                ),
                &make_primitive(&poly_ring, &h).0
            )
        );
        assert!(
            poly_ring.divides(
                &poly_ring.prod(
                    power_decomp
                        .iter()
                        .filter(|(_, k)| k % 2 == 0)
                        .map(|(poly, k)| poly_ring.pow(poly.clone(), k / 2))
                ),
                &make_primitive(&poly_ring, &g).0
            )
        );
    }
}

#[test]
fn test_poly_gcd_number_field() {
    feanor_tracing::DelayedLogger::init_test();
    let field = test_field();
    let poly_ring = DensePolyRing::new(field, "X");
    let zeta = RingElementWrapper::new(
        &poly_ring,
        poly_ring.inclusion().map(poly_ring.base_ring().canonical_gen()),
    );
    let irred_polys = poly_ring.with_wrapped_indeterminate(|X| {
        [
            X - &zeta,
            X + &zeta,
            X.pow_ref(2) + zeta.pow_ref(1) * X + 100,
            &zeta * X.pow_ref(3) - X + 1,
        ]
    });
    let poly = |powers: [usize; 4]| {
        poly_ring.prod(
            powers
                .iter()
                .zip(irred_polys.iter())
                .map(|(e, f)| poly_ring.pow(f.clone(), *e)),
        )
    };

    assert_el_eq!(
        &poly_ring,
        poly_ring.normalize(poly([1, 0, 0, 0])).0,
        poly_gcd_number_field(&poly_ring, &poly([1, 0, 1, 0]), &poly([1, 0, 0, 1]))
    );
    assert_el_eq!(
        &poly_ring,
        poly_ring.normalize(poly([1, 0, 1, 0])).0,
        poly_gcd_number_field(&poly_ring, &poly([1, 1, 1, 0]), &poly([1, 0, 1, 1]))
    );
    assert_el_eq!(
        &poly_ring,
        poly_ring.normalize(poly([1, 0, 2, 0])).0,
        poly_gcd_number_field(&poly_ring, &poly([1, 1, 3, 0]), &poly([3, 0, 2, 0]))
    );
    assert_el_eq!(
        &poly_ring,
        poly_ring.normalize(poly([1, 0, 0, 5])).0,
        poly_gcd_number_field(&poly_ring, &poly([2, 1, 3, 5]), &poly([1, 0, 0, 7]))
    );
}

#[test]
fn test_poly_power_decomposition_number_field_degenerate_reduction() {
    feanor_tracing::DelayedLogger::init_test();
    let field = test_field();
    let poly_ring = DensePolyRing::new(field, "X");
    let ring = poly_ring.base_ring();
    let f = poly_ring.from_terms([
        (
            ring.inclusion()
                .compose(QQ.inclusion())
                .map(ZZbig.prod(LARGE_PRIMES[..6].iter().map(|x| int_cast(*x, ZZbig, ZZi64)))),
            2,
        ),
        (ring.one(), 0),
    ]);
    let power_decomposition = poly_power_decomposition_number_field(&poly_ring, &f);
    assert_eq!(1, power_decomposition.len());
    assert_el_eq!(&poly_ring, poly_ring.normalize(f).0, power_decomposition[0].0);
}

#[test]
fn random_test_poly_gcd_number_field() {
    feanor_tracing::DelayedLogger::init_test();
    let field = test_field();
    let poly_ring = DensePolyRing::new(field, "X");
    let ring = poly_ring.base_ring();
    let mut rng = oorandom::Rand64::new(1);
    let bound = ZZbig.int_hom().map(500);
    let mut random_poly_of_deg = |deg: usize| {
        poly_ring.from_terms((0..=deg).map(|i| {
            (
                ring.from_canonical_basis((0..ring.rank()).map(|_| {
                    QQ.inclusion()
                        .map(QQ.base_ring().get_uniformly_random(&bound, || rng.rand_u64()))
                })),
                i,
            )
        }))
    };

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let f = random_poly_of_deg(14);
        let g = random_poly_of_deg(12);
        let h = random_poly_of_deg(10);
        let lhs = poly_ring.mul_ref(&f, &h);
        let rhs = poly_ring.mul_ref(&g, &h);
        let gcd = make_primitive(&poly_ring, &poly_gcd_number_field(&poly_ring, &lhs, &rhs)).0;

        assert!(poly_ring.divides(&lhs, &gcd));
        assert!(poly_ring.divides(&rhs, &gcd));
        assert!(poly_ring.divides(&gcd, &make_primitive(&poly_ring, &h).0));
    }
}
