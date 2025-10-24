
use std::mem::swap;
use std::sync::atomic::Ordering;

use atomicbox::AtomicOptionBox;

use crate::computation::{ComputationController, DontObserve, ShortCircuitingComputation, ShortCircuitingComputationHandle};
use crate::delegate::{UnwrapHom, WrapHom};
use crate::reduce_lift::lift_poly_eval::{LiftPolyEvalRing, LiftPolyEvalRingReductionMap};
use crate::divisibility::{DivisibilityRingStore, Domain};
use crate::pid::*;
use crate::algorithms::eea::signed_lcm;
use crate::rings::field::{AsField, AsFieldBase};
use crate::rings::fraction::FractionFieldStore;
use crate::rings::poly::*;
use crate::rings::finite::*;
use crate::pid::EuclideanRingStore;
use crate::seq::VectorFn;
use crate::specialization::FiniteRingOperation;
use crate::integer::*;
use crate::rings::rational::*;
use crate::homomorphism::*;
use crate::ring::*;
use crate::rings::poly::dense_poly::DensePolyRing;

///
/// Computes the resultant of two polynomials `f` and `g` over a finite field.
/// 
/// Usually you should use [`ComputeResultantRing::resultant()`], unless you
/// are implementing said method for a custom ring.
/// 
#[stability::unstable(feature = "enable")]
pub fn resultant_finite_field<P>(ring: P, mut f: El<P>, mut g: El<P>) -> El<<P::Type as RingExtension>::BaseRing>
    where P: RingStore + Copy,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Domain + FiniteRing
{
    let base_ring = ring.base_ring();
    if ring.is_zero(&g) || ring.is_zero(&f) {
        return base_ring.zero();
    }
    let mut scale = base_ring.one();
    if ring.degree(&g).unwrap() < ring.degree(&f).unwrap() {
        if ring.degree(&g).unwrap() % 2 != 0 && ring.degree(&f).unwrap() % 2 != 0 {
            base_ring.negate_inplace(&mut scale);
        }
        swap(&mut f, &mut g);
    }

    while ring.degree(&f).unwrap_or(0) >= 1 {
        assert!(ring.degree(&g).unwrap() >= ring.degree(&f).unwrap());
        let deg_g = ring.degree(&g).unwrap();
        let r = ring.euclidean_rem(g, &f);
        g = r;
        base_ring.mul_assign(&mut scale, base_ring.pow(base_ring.clone_el(ring.lc(&f).unwrap()), deg_g - ring.degree(&g).unwrap_or(0)));

        swap(&mut f, &mut g);
        if ring.degree(&g).unwrap() % 2 != 0 && ring.degree(&f).unwrap_or(0) % 2 != 0 {
            base_ring.negate_inplace(&mut scale);
        }
    }

    if ring.is_zero(&f) {
        return base_ring.zero();
    } else {
        let mut result = base_ring.clone_el(&ring.coefficient_at(&f, 0));
        result = base_ring.pow(result, ring.degree(&g).unwrap());
        base_ring.mul_assign(&mut result, scale);
        return result;
    }
}

///
/// Trait for rings that support computing resultants of polynomials
/// over the ring.
/// 
#[stability::unstable(feature = "enable")]
pub trait ComputeResultantRing: RingBase {

    ///
    /// Computes the resultant of `f` and `g` over the base ring, taking
    /// a controller to control the performed computation.
    /// 
    /// See also [`ComputeResultantRing::resultant()`].
    /// 
    fn resultant_with_controller<P, Controller>(poly_ring: P, f: El<P>, g: El<P>, controller: Controller) -> Self::Element
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>,
            Controller: ComputationController;
    
    ///
    /// Computes the resultant of `f` and `g` over the base ring.
    /// 
    /// The resultant is the determinant of the linear map
    /// ```text
    ///   R[X]_deg(g)  x  R[X]_deg(f)  ->  R[X]_deg(fg),
    ///        a       ,       b       ->    af + bg
    /// ```
    /// where `R[X]_d` refers to the vector space of polynomials in `R[X]` of degree
    /// less than `d`.
    /// 
    /// # Example
    /// ```rust
    /// use feanor_math::assert_el_eq;
    /// use feanor_math::ring::*;
    /// use feanor_math::integer::*;
    /// use feanor_math::rings::poly::dense_poly::DensePolyRing;
    /// use feanor_math::rings::poly::*;
    /// use feanor_math::algorithms::resultant::*;
    /// let ZZ = BigIntRing::RING;
    /// let ZZX = DensePolyRing::new(ZZ, "X");
    /// let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) - 2 * X + 1]);
    /// // the discrimiant is the resultant of f and f'
    /// let discriminant = <_ as ComputeResultantRing>::resultant(&ZZX, ZZX.clone_el(&f), derive_poly(&ZZX, &f));
    /// assert_el_eq!(ZZ, ZZ.zero(), discriminant);
    /// ```
    /// 
    fn resultant<P>(poly_ring: P, f: El<P>, g: El<P>) -> Self::Element
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        Self::resultant_with_controller(poly_ring, f, g, DontObserve)
    }
}

impl<R: ?Sized + LiftPolyEvalRing + Domain + SelfIso> ComputeResultantRing for R {

    default fn resultant_with_controller<P, Controller>(ring: P, f: El<P>, g: El<P>, controller: Controller) -> El<<P::Type as RingExtension>::BaseRing>
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = R>,
            Controller: ComputationController
    {
        struct ComputeResultant<P, Controller>
            where P: RingStore + Copy,
                P::Type: PolyRing,
                <<P::Type as RingExtension>::BaseRing as RingStore>::Type: LiftPolyEvalRing + Domain + SelfIso,
                Controller: ComputationController
        {
            ring: P,
            f: El<P>,
            g: El<P>,
            controller: Controller
        }
        impl<P, Controller> FiniteRingOperation<<<P::Type as RingExtension>::BaseRing as RingStore>::Type> for ComputeResultant<P, Controller>
            where P: RingStore + Copy,
                P::Type: PolyRing,
                <<P::Type as RingExtension>::BaseRing as RingStore>::Type: LiftPolyEvalRing + Domain + SelfIso,
                Controller: ComputationController
        {
            type Output = El<<P::Type as RingExtension>::BaseRing>;

            fn execute(self) -> Self::Output
                where <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing
            {
                let new_poly_ring = DensePolyRing::new(AsField::from(AsFieldBase::promise_is_perfect_field(self.ring.base_ring())), "X");
                let hom = new_poly_ring.lifted_hom(&self.ring, WrapHom::new(new_poly_ring.base_ring().get_ring()));
                let result = resultant_finite_field(&new_poly_ring, hom.map(self.f), hom.map(self.g));
                return UnwrapHom::new(new_poly_ring.base_ring().get_ring()).map(result);
            }

            fn fallback(self) -> Self::Output {
                let ring_ref = &self.ring;
                let f_ref = &self.f;
                let g_ref = &self.g;
                let base_ring = ring_ref.base_ring();
                if ring_ref.is_zero(f_ref) || ring_ref.is_zero(g_ref) {
                    return base_ring.zero();
                }
                self.controller.run_computation(format_args!("resultant_local(ldeg={}, rdeg={})", ring_ref.degree(f_ref).unwrap(), ring_ref.degree(g_ref).unwrap()), |controller| {
                    let coeff_bound_f_ln = ring_ref.terms(f_ref).map(|(c, _)| base_ring.get_ring().ln_pseudo_norm(c)).max_by(f64::total_cmp).unwrap();
                    let coeff_bound_g_ln = ring_ref.terms(g_ref).map(|(c, _)| base_ring.get_ring().ln_pseudo_norm(c)).max_by(f64::total_cmp).unwrap();
                    let ln_max_norm = coeff_bound_f_ln * ring_ref.degree(g_ref).unwrap() as f64 + 
                        coeff_bound_g_ln * ring_ref.degree(f_ref).unwrap() as f64 + 
                        // this is just an estimate on the number of terms we sum up: for each column belonging to `f`, there are `deg(f)` nonzero entries, and
                        // we have `deg(g)` such columns, thus the number of terms is bounded by `deg(f)^deg(g)`; similarly for `g` 
                        base_ring.get_ring().ln_pseudo_norm(&base_ring.int_hom().map(ring_ref.degree(f_ref).unwrap() as i32)) * ring_ref.degree(g_ref).unwrap() as f64 +
                        base_ring.get_ring().ln_pseudo_norm(&base_ring.int_hom().map(ring_ref.degree(g_ref).unwrap() as i32)) * ring_ref.degree(f_ref).unwrap() as f64;

                    let work_locally = base_ring.get_ring().init_reduce_lift(ln_max_norm);
                    let work_locally_ref = &work_locally;
                    let count = base_ring.get_ring().prime_ideal_count(&work_locally);
                    log_progress!(controller, "({})", count);
                    let resultants = (0..count).map(|_| AtomicOptionBox::none()).collect::<Vec<_>>();
                    let resultants_ref = &resultants;

                    ShortCircuitingComputation::<(), _>::new().handle(controller).join_many((0..count).map_fn(|i| move |handle: ShortCircuitingComputationHandle<_, _>| {
                        let embedding = LiftPolyEvalRingReductionMap::new(base_ring.get_ring(), work_locally_ref, i);
                        let new_poly_ring = DensePolyRing::new(embedding.codomain(), "X");
                        let poly_ring_embedding = new_poly_ring.lifted_hom(ring_ref, &embedding);
                        let local_f = poly_ring_embedding.map_ref(f_ref);
                        let local_g = poly_ring_embedding.map_ref(g_ref);
                        let local_resultant = <_ as ComputeResultantRing>::resultant(&new_poly_ring, local_f, local_g);
                        _ = resultants_ref[i].swap(Some(Box::new(local_resultant)), Ordering::SeqCst);
                        log_progress!(handle, ".");
                        return Ok(None);
                    }));

                    let resultants = resultants.into_iter().map(|r| *r.swap(None, Ordering::SeqCst).unwrap()).collect::<Vec<_>>();
                    return base_ring.get_ring().lift_combine(&work_locally, &resultants);
                })
            }
        }
        R::specialize(ComputeResultant { ring, f, g, controller })
    }
}

impl<I> ComputeResultantRing for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing 
{
    fn resultant_with_controller<P, Controller>(ring: P, f: El<P>, g: El<P>, controller: Controller) -> El<<P::Type as RingExtension>::BaseRing>
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>,
            Controller: ComputationController
    {
        if ring.is_zero(&g) || ring.is_zero(&f) {
            return ring.base_ring().zero();
        }
        let QQ = ring.base_ring();
        let ZZ = QQ.base_ring();
        let f_deg = ring.degree(&f).unwrap();
        let g_deg = ring.degree(&g).unwrap();
        let f_den_lcm = ring.terms(&f).map(|(c, _)| ZZ.clone_el(QQ.get_ring().den(c))).fold(ZZ.one(), |a, b| signed_lcm(a, b, ZZ));
        let g_den_lcm = ring.terms(&g).map(|(c, _)| ZZ.clone_el(QQ.get_ring().den(c))).fold(ZZ.one(), |a, b| signed_lcm(a, b, ZZ));
        let ZZX = DensePolyRing::new(ZZ, "X");
        let f_int = ZZX.from_terms(ring.terms(&f).map(|(c, i)| { let (a, b) = (QQ.get_ring().num(c), QQ.get_ring().den(c)); (ZZ.checked_div(&ZZ.mul_ref(&f_den_lcm, a), b).unwrap(), i) }));
        let g_int = ZZX.from_terms(ring.terms(&g).map(|(c, i)| { let (a, b) = (QQ.get_ring().num(c), QQ.get_ring().den(c)); (ZZ.checked_div(&ZZ.mul_ref(&f_den_lcm, a), b).unwrap(), i) }));
        let result_unscaled = <_ as ComputeResultantRing>::resultant_with_controller(&ZZX, f_int, g_int, controller);
        return QQ.from_fraction(result_unscaled, ZZ.mul(ZZ.pow(f_den_lcm, g_deg), ZZ.pow(g_den_lcm, f_deg)));
    }
}

#[cfg(test)]
use crate::rings::multivariate::multivariate_impl::MultivariatePolyRingImpl;
#[cfg(test)]
use crate::rings::multivariate::*;
#[cfg(test)]
use crate::algorithms::buchberger::buchberger;
#[cfg(test)]
use crate::integer::BigIntRing;
#[cfg(test)]
use crate::algorithms::poly_gcd::PolyTFracGCDRing;

#[test]
#[allow(deprecated)]
fn test_resultant_local_polynomial() {
    let ZZ = BigIntRing::RING;
    let QQ = RationalField::new(ZZ);
    static_assert_impls!(RationalFieldBase<BigIntRing>: PolyTFracGCDRing);
    // we eliminate `Y`, so add it as the outer indeterminate
    let QQX = DensePolyRing::new(&QQ, "X");
    let QQXY = DensePolyRing::new(&QQX, "Y");
    let ZZ_to_QQ = QQ.int_hom();

    // 1 + X^2 + 2 Y + (1 + X) Y^2
    let f= QQXY.from_terms([
        (vec![(1, 0), (1, 2)], 0),
        (vec![(2, 0)], 1),
        (vec![(1, 0), (1, 1)], 2)
    ].into_iter().map(|(v, i)| (QQX.from_terms(v.into_iter().map(|(c, j)| (ZZ_to_QQ.map(c), j))), i)));

    // 3 + X + (2 + X) Y + (1 + X + X^2) Y^2
    let g = QQXY.from_terms([
        (vec![(3, 0), (1, 1)], 0),
        (vec![(2, 0), (1, 1)], 1),
        (vec![(1, 0), (1, 1), (1, 2)], 2)
    ].into_iter().map(|(v, i)| (QQX.from_terms(v.into_iter().map(|(c, j)| (ZZ_to_QQ.map(c), j))), i)));

    let actual = <_ as ComputeResultantRing>::resultant(&QQXY, QQXY.clone_el(&f), QQXY.clone_el(&g));
    let [expected] = QQX.with_wrapped_indeterminate(|X| [X.pow_ref(8) + 2 * X.pow_ref(7) + 3 * X.pow_ref(6) - 5 * X.pow_ref(5) - 10 * X.pow_ref(4) - 7 * X.pow_ref(3) + 8 * X.pow_ref(2) + 8 * X + 4]);
    assert_el_eq!(&QQX, &expected, &actual);

    let QQYX = MultivariatePolyRingImpl::new(&QQ, 2);
    // reverse the order of indeterminates, so that we indeed eliminate `Y`
    let [f, g] = QQYX.with_wrapped_indeterminates(|[Y, X]| [ 1 + X.pow_ref(2) + 2 * Y + (1 + X) * Y.pow_ref(2), 3 + X + (2 + X) * Y + (1 + X + X.pow_ref(2)) * Y.pow_ref(2) ]);

    let gb = buchberger::<_, _>(&QQYX, vec![f, g], Lex);
    let expected = gb.into_iter().filter(|poly| QQYX.appearing_indeterminates(&poly).len() == 1).collect::<Vec<_>>();
    assert!(expected.len() == 1);
    let expected = QQX.normalize(QQX.from_terms(QQYX.terms(&expected[0]).map(|(c, m)| (QQ.clone_el(c), QQYX.exponent_at(m, 1)))));

    assert_el_eq!(QQX, expected, actual);
}

#[test]
fn test_resultant_local_integer() {
    let ZZ = BigIntRing::RING;
    let ZZX = DensePolyRing::new(ZZ, "X");

    let [f, g] = ZZX.with_wrapped_indeterminate(|X| [
        X.pow_ref(32) + 1,
        X.pow_ref(2) - X - 1
    ]);
    assert_el_eq!(ZZ, ZZ.int_hom().map(4870849), <_ as ComputeResultantRing>::resultant(&ZZX, f, g));

    let [f, g] = ZZX.with_wrapped_indeterminate(|X| [
        X.pow_ref(4) - 2 * X + 2,
        X.pow_ref(64) + 1
    ]);
    assert_el_eq!(ZZ, ZZ.parse("381380816531458621441", 10).unwrap(), <_ as ComputeResultantRing>::resultant(&ZZX, f, g));

    let [f, g] = ZZX.with_wrapped_indeterminate(|X| [
        X.pow_ref(32) - 2 * X.pow_ref(8) + 2,
        X.pow_ref(512) + 1
    ]);
    assert_el_eq!(ZZ, ZZ.pow(ZZ.parse("381380816531458621441", 10).unwrap(), 8), <_ as ComputeResultantRing>::resultant(&ZZX, f, g));
}

#[test]
#[ignore]
fn test_resultant_large() {
    let ZZ = BigIntRing::RING;
    let ZZX = DensePolyRing::new(ZZ, "X");

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(4) - 2 * X + 2]);
    let g = ZZX.from_terms([(ZZ.one(), 1 << 14), (ZZ.one(), 0)]);
    println!("start");
    let result = BigIntRingBase::resultant(&ZZX, f, g);
    println!("{}", ZZ.formatted_el(&result))
}