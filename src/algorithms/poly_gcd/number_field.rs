use std::sync::atomic::AtomicBool;

use tracing::instrument;

use crate::PROBABILISTIC_REPETITIONS;
use crate::algorithms::hensel::HenselLift;
use crate::algorithms::interpolate::product_except_one;
use crate::algorithms::poly_factor::FactorPolyField;
use crate::algorithms::poly_gcd::gcd_lift::LiftUnsuccessful;
use crate::algorithms::poly_gcd::power_decomposition_lift::*;
use crate::algorithms::poly_gcd::*;
use crate::algorithms::poly_root::poly_root;
use crate::algorithms::primelist::large_prime_fields;
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

#[stability::unstable(feature = "enable")]
#[derive(Debug, Copy, Clone)]
pub enum QuotientAtError {
    Ramified,
    ReductionNotWellDefined,
}

#[stability::unstable(feature = "enable")]
pub type ResidueField = GaloisField<AsField<FreeAlgebraImpl<AsField<Zn64B>, Vec<El<AsField<Zn64B>>>>>>;

#[stability::unstable(feature = "enable")]
pub type ResidueFieldPolyRing = DensePolyRing<ResidueField>;

#[stability::unstable(feature = "enable")]
pub struct ResidueFieldsAtPrime {
    FqXs: Vec<ResidueFieldPolyRing>,
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
    LambdaHom::new(number_field, galois_ring, move |K, GR, x| {
        GR.from_canonical_basis_extended(K.wrt_canonical_basis(x).iter().map(|c| mod_p.map(c)))
    })
}

#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn quotient_at<K, I>(number_field: K, Fp: AsField<Zn64B>) -> Result<ResidueFieldsAtPrime, QuotientAtError>
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
    let error = AtomicBool::new(false);
    let generating_poly = number_field.generating_poly(&FpX, QQ_to_Zpe_hom(QQ, Fp, &error));
    if error.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(QuotientAtError::ReductionNotWellDefined);
    }

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

#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_in_quotients<'b, P, I>(
    KX: P,
    poly: &El<P>,
    residue_fields: &'b ResidueFieldsAtPrime,
) -> Result<Vec<El<ResidueFieldPolyRing>>, ()>
where
    P: RingStore + Copy,
    P::Ring: PolyRing,
    BaseRingBase<P>: Field + FreeAlgebra,
    BaseRingStore<BaseRingStore<P>>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    let mut result = Vec::new();
    let K = KX.base_ring();
    let Fp = residue_fields.FqXs[0].base_ring().base_ring();
    assert!(Fp.get_ring() == residue_fields.FqXs[0].base_ring().base_ring().get_ring());

    for FqX in &residue_fields.FqXs {
        let Fq = FqX.base_ring();
        let error = AtomicBool::new(false);
        let poly_mod_p = FqX.lifted_hom(KX, K_to_GR_hom(K, Fq, &error)).map_ref(poly);
        if error.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(());
        }
        result.push(poly_mod_p);
    }
    return Ok(result);
}

#[stability::unstable(feature = "enable")]
pub type ResidueRing = FreeAlgebraImpl<ZnGB<BigIntRing>, Vec<El<ZnGB<BigIntRing>>>>;

#[stability::unstable(feature = "enable")]
pub type ResidueRingPolyRing = DensePolyRing<ResidueRing>;

#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn reconstruct_number_field_el<K, I>(
    number_field: K,
    residue_rings: &[&ResidueRingPolyRing],
    residue_values: &[El<ResidueRing>],
) -> El<K>
where
    K: RingStore,
    K::Ring: FreeAlgebra + Field,
    BaseRingStore<K>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    assert_eq!(residue_rings.len(), residue_values.len());
    let QQ = number_field.base_ring();
    let ZZ = QQ.base_ring();
    let GR = residue_rings[0].base_ring();
    let Zpe = GR.base_ring();
    let ZpeX = DensePolyRing::new(Zpe, "X");
    assert!(
        residue_rings
            .iter()
            .all(|GRX| GRX.base_ring().base_ring().get_ring() == Zpe.get_ring())
    );

    // compute data necessary for inverse CRT
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
    let error = AtomicBool::new(false);
    debug_assert_el_eq!(
        &ZpeX,
        &complete_product,
        &number_field.generating_poly(&ZpeX, QQ_to_Zpe_hom(QQ, Zpe, &error))
    );
    debug_assert!(!error.load(std::sync::atomic::Ordering::Relaxed));
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

    // now apply inverse CRT to get the value over ZpeX
    let combined = <_ as RingStore>::sum(
        &ZpeX,
        residue_rings.iter().enumerate().map(|(i, GRX)| {
            let unreduced_result = ZpeX.mul_ref_snd(
                GRX.base_ring().poly_repr(&ZpeX, &residue_values[i], Zpe.identity()),
                &unit_vectors[i],
            );
            ZpeX.div_rem_monic(unreduced_result, &complete_product).1
        }),
    );
    for (i, GRX) in residue_rings.iter().enumerate() {
        debug_assert_el_eq!(
            GRX.base_ring(),
            &residue_values[i],
            &ZpeX.evaluate(&combined, &GRX.base_ring().canonical_gen(), GRX.base_ring().inclusion())
        );
    }
    // now lift the polynomial modulo `p^e` to the rationals
    let result = number_field.from_canonical_basis((0..number_field.rank()).map(|i| {
        let (num, den) = balanced_rational_reconstruction(Zpe, ZpeX.coefficient_at(&combined, i).clone());
        return QQ.div(
            &QQ.inclusion().map(int_cast(num, ZZ, Zpe.integer_ring())),
            &QQ.inclusion().map(int_cast(den, ZZ, Zpe.integer_ring())),
        );
    }));
    return result;
}

struct PolyPowerDecompositionLift {
    gen_poly_lifter: HenselLift<DensePolyRing<ZnGB<BigIntRing>>>,
    power_factor_lifters: Vec<HenselLift<ResidueRingPolyRing>>,
    exponents: Vec<usize>,
    prime: El<BigIntRing>,
}

impl PolyPowerDecompositionLift {
    #[instrument(skip_all, level = "trace")]
    fn new(
        residue_fields: ResidueFieldsAtPrime,
        power_decompositions: Vec<Vec<(El<ResidueFieldPolyRing>, usize)>>,
    ) -> Result<Self, BadPrime> {
        let Fp = residue_fields.FqXs[0].base_ring().base_ring().clone();
        let FpX = DensePolyRing::new(Fp, "X");
        let Zp = ZnGB::new(ZZbig, int_cast(*Fp.modulus(), ZZbig, ZZi64));
        let ZpX = DensePolyRing::new(Zp.clone(), "X");
        let hom = ZnReductionMap::new(&Fp, &Zp).unwrap();
        let gen_poly_factorization = residue_fields
            .FqXs
            .iter()
            .map(|FqX| FqX.base_ring().generating_poly(&FpX, FpX.base_ring().identity()))
            .collect::<Vec<_>>();
        let gen_poly_lifter = HenselLift::new(FpX, gen_poly_factorization).unwrap();
        let mut exponents = None;
        let mut lifters = Vec::new();
        for (mut power_decomposition, FqX) in power_decompositions.into_iter().zip(residue_fields.FqXs) {
            let Fq = FqX.base_ring().clone();
            power_decomposition.sort_unstable_by_key(|(_, e)| *e);
            let expected_exponents = power_decomposition.iter().map(|(_, e)| *e).collect::<Vec<_>>();
            if let Some(exponents) = &exponents {
                if &expected_exponents != exponents {
                    return Err(BadPrime);
                }
            } else {
                exponents = Some(expected_exponents);
            }
            let factors = power_decomposition
                .into_iter()
                .map(|(f, i)| FqX.pow(f, i))
                .collect::<Vec<_>>();
            let modulus = Fq.generating_poly(&ZpX, &hom);
            let GR = FreeAlgebraImpl::new(
                Zp.clone(),
                Fq.rank(),
                (0..Fq.rank())
                    .map(|i| Zp.negate(ZpX.coefficient_at(&modulus, i).clone()))
                    .collect::<Vec<_>>(),
            );
            let GRX = DensePolyRing::new(GR.clone(), "X");
            lifters.push(
                HenselLift::new(FqX, factors)
                    .unwrap()
                    .change_ring(GRX, GR.lifted_hom(Fq, &hom).unwrap()),
            );
        }
        return Ok(Self {
            exponents: exponents.unwrap(),
            gen_poly_lifter: gen_poly_lifter.change_ring(ZpX, hom),
            power_factor_lifters: lifters,
            prime: Zp.modulus().clone(),
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
        let Zpe = ZnGB::new(ZZbig, ZZbig.pow(self.prime.clone(), lift_to_degree));
        let ZpeX = DensePolyRing::new(Zpe.clone(), "X");
        let K = KX.base_ring();
        let QQ = K.base_ring();
        let error = AtomicBool::new(false);
        {
            let lifter = &mut self.gen_poly_lifter;
            let target_mod_pe = K.generating_poly(&ZpeX, QQ_to_Zpe_hom(QQ, &Zpe, &error));
            let mod_pe = Zpe.can_hom(&ZZbig).unwrap();
            take_mut::take(lifter, |lifter| {
                lifter.lift_to(lift_to_degree, ZpeX.clone(), &target_mod_pe, |old_base_ring, _, x| {
                    mod_pe.map(old_base_ring.smallest_lift(x.clone()))
                })
            });
        }
        for (lifter, modulus) in self
            .power_factor_lifters
            .iter_mut()
            .zip(self.gen_poly_lifter.factorization())
        {
            let old_ring = lifter.poly_ring();
            let old_base_ring = old_ring.base_ring();
            let GR = FreeAlgebraImpl::new(
                Zpe.clone(),
                old_base_ring.rank(),
                (0..old_base_ring.rank())
                    .map(|i| Zpe.negate(ZpeX.coefficient_at(modulus, i).clone()))
                    .collect(),
            );
            let GRX = DensePolyRing::new(GR, "X");
            let target_mod_pe = GRX
                .lifted_hom(KX, K_to_GR_hom(K, GRX.base_ring(), &error))
                .map_ref(target);
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
        debug_assert!(!error.load(std::sync::atomic::Ordering::Relaxed));

        let residue_rings = self
            .power_factor_lifters
            .iter()
            .map(|lifter| lifter.poly_ring())
            .collect::<Vec<_>>();
        let factor_count = self.power_factor_lifters[0].factorization().count();
        let factor_degrees = self.power_factor_lifters[0]
            .factorization()
            .map(|f| self.power_factor_lifters[0].poly_ring().degree(f).unwrap())
            .collect::<Vec<_>>();
        let lifted_factorization = (0..factor_count)
            .map(|i| {
                KX.from_terms((0..=factor_degrees[i]).map(|j| {
                    (
                        reconstruct_number_field_el(
                            KX.base_ring(),
                            &residue_rings,
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

        assert_eq!(self.exponents.len(), lifted_factorization.len());
        if KX.eq_el(&KX.prod(lifted_factorization.iter().cloned()), target) {
            return lifted_factorization
                .into_iter()
                .zip(self.exponents.iter())
                .map(|(f, i)| poly_root(KX, &f, *i).map(|f| (f, *i)).ok_or(LiftUnsuccessful))
                .collect();
        } else {
            return Err(LiftUnsuccessful);
        }
    }
}

/// Computes the power decomposition of polynomials `f, g in K[X]`.
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
            let residue_fields = quotient_at(K, Fp).ok()?;
            let poly_mod_p = poly_in_quotients(KX, &poly, &residue_fields).ok()?;
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

#[test]
fn test_poly_power_decomposition_number_field() {
    feanor_tracing::DelayedLogger::init_test();
    let QQ = RationalField::new(ZZbig);
    let ring = NumberField::from(NumberFieldBase::create(AsField::from(
        AsFieldBase::promise_is_field(FreeAlgebraImpl::new(&QQ, 4, [QQ.one()])).unwrap(),
    )));
    let poly_ring = DensePolyRing::new(ring, "X");
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
    let QQ = RationalField::new(ZZbig);
    let ring = NumberField::from(NumberFieldBase::create(AsField::from(
        AsFieldBase::promise_is_field(FreeAlgebraImpl::new(&QQ, 4, [QQ.one()])).unwrap(),
    )));
    let poly_ring = DensePolyRing::new(ring, "X");
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
