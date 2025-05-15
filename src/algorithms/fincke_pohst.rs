use crate::ring::*;
use crate::homomorphism::Homomorphism;
use crate::integer::*;
use crate::field::*;
use crate::matrix::*;
use crate::algorithms::lll::LLLRealField;
use crate::ordered::OrderedRingStore;
use crate::algorithms::matmul::ComputeInnerProduct;

///
/// Uses the Fincke-Pohst algorithm to find integer points close to `target`, where
/// "closeness" is measured according to the norm defined by the given quadratic form.
/// 
/// More concretely, this function calls `for_point` for each point `x` for which
/// `(x - target)^T Q (x - target) < radius_sqr`, where `Q` is the given quadratic form.
/// If `for_point` returns a value, this is used as the new `radius_sqr` for the remainder
/// of the computation. In particular, when using `fincke_pohst()` to find shortest or
/// closest vectors, this can be used to speed up the remaining computation once a solution
/// has been found.
/// 
/// Note that using [`crate::rings::float_real::Real64`] for `RR` will result in a computation
/// with floating point numbers, and thus points of squared distance roughly `radius_sqr` may
/// not be selected accurately.
/// 
#[stability::unstable(feature = "enable")]
pub fn fincke_pohst<I, R, V, F>(ZZ: I, RR: R, quadratic_form: Submatrix<V, El<R>>, target: &[El<R>], radius_sqr: El<R>, mut for_point: F)
    where I: RingStore + Copy,
        I::Type: IntegerRing,
        R: RingStore + Copy,
        R::Type: LLLRealField<I::Type>,
        V: AsPointerToSlice<El<R>>,
        F: FnMut(&[El<I>]) -> Option<El<R>>
{
    let n = quadratic_form.row_count();
    assert_eq!(n, quadratic_form.col_count());
    assert_eq!(n, target.len());
    let one = ZZ.one();
    let update = |x: &mut El<I>, center: &El<I>| {
        *x = ZZ.add_ref_fst(center, ZZ.sub_ref(center, x));
        if ZZ.is_geq(x, center) {
            ZZ.add_assign_ref(x, &one);
        }
    };

    let mut radius_square = radius_sqr;
    let mut i = 0;
    let mut current = (0..n).map(|_| ZZ.zero()).collect::<Vec<_>>();
    let mut centers = (0..n).map(|_| ZZ.zero()).collect::<Vec<_>>();
    // the inner product of the `i`-th column of `quadratic_form` with `(current[..i] - target[..i], 0)`
    let mut linear_components = (0..n).map(|_| RR.zero()).collect::<Vec<_>>();
    // squared norm w.r.t. `quadratic_form` of `(current[..i] - target[..i], 0)`
    let mut norms_square = (0..n).map(|_| RR.zero()).collect::<Vec<_>>();

    linear_components[i] = RR.zero();
    norms_square[i] = RR.zero();
    let center_int = RR.get_ring().round_to_integer(&target[i], ZZ.get_ring());
    current[i] = ZZ.clone_el(&center_int);
    centers[i] = center_int;

    'visit_enum_tree_node: loop {
        // since we don't consider whether `center` over-or underestimates the real value for
        // center, we have to consider both ends of the interval; we do this using a for loop
        for _ in 0..2 {
            let x = RR.sub_ref_snd(
                RR.get_ring().from_integer(&current[i], ZZ.get_ring()),
                &target[i]
            );
            let norm_square = RR.add_ref_fst(
                &norms_square[i],
                RR.mul_ref_fst(
                    &x,
                    RR.add(
                        RR.mul_ref(quadratic_form.at(i, i), &x),
                        RR.int_hom().mul_ref_map(&linear_components[i], &2)
                    )
                )
            );
    
            if RR.is_lt(&norm_square, &radius_square) {
                i += 1;
                if i == n {
                    if let Some(new_radius_square) = for_point(&current) {
                        radius_square = new_radius_square;
                    }
                    i -= 1;
                    update(&mut current[i], &centers[i]);
                } else {
                    linear_components[i] = <_ as ComputeInnerProduct>::inner_product_ref_fst(RR.get_ring(), (0..i).map(|j| (
                        quadratic_form.at(j, i),
                        RR.sub_ref_snd(
                            RR.get_ring().from_integer(&current[j], ZZ.get_ring()),
                            &target[j]
                        )
                    )));
                    let center = RR.sub_ref_fst(
                        &target[i],
                        RR.div(&linear_components[i], quadratic_form.at(i, i))
                    );
                    let center_int = RR.get_ring().round_to_integer(&center, ZZ.get_ring());
                    current[i] = ZZ.clone_el(&center_int);
                    centers[i] = center_int;
                    norms_square[i] = norm_square;
                }
                continue 'visit_enum_tree_node;
            }
            update(&mut current[i], &centers[i]);
        }
        if i == 0 {
            return;
        } else {
            i -= 1;
            update(&mut current[i], &centers[i]);
        }
    }
}

#[cfg(test)]
use crate::rings::float_real::*;
#[cfg(test)]
use crate::primitive_int::StaticRing;

#[test]
fn test_fincke_pohst_2d() {
    let ZZ = StaticRing::<i64>::RING;
    let RR = Real64::RING;

    let quadratic_form = [vec![1., 0.], vec![0., 1.]];
    let mut result = Vec::new();
    fincke_pohst(&ZZ, &RR, Submatrix::from_2d(&quadratic_form), &[0., 0.], 2.001, |point| { result.push(point.to_owned()); None });
    assert_eq!(vec![vec![0, 0], vec![0, 1], vec![0, -1], vec![1, 0], vec![1, 1], vec![1, -1], vec![-1, 0], vec![-1, 1], vec![-1, -1]], result);
    
    let quadratic_form = [vec![1., 0.], vec![0., 1.]];
    let mut result = Vec::new();
    fincke_pohst(&ZZ, &RR, Submatrix::from_2d(&quadratic_form), &[0.51, 0.51], 0.6, |point| { result.push(point.to_owned()); None });
    assert_eq!(vec![vec![1, 1], vec![1, 0], vec![0, 1], vec![0, 0]], result);
    
    let quadratic_form = [vec![1., 0.], vec![0., 1.]];
    let mut result = Vec::new();
    fincke_pohst(&ZZ, &RR, Submatrix::from_2d(&quadratic_form), &[10.75, 15.1], 0.3, |point| { result.push(point.to_owned()); None });
    assert_eq!(vec![vec![11, 15]], result);
    
    let quadratic_form = [vec![2., 1.], vec![1., 2.]];
    let mut result = Vec::new();
    fincke_pohst(&ZZ, &RR, Submatrix::from_2d(&quadratic_form), &[0.22, 0.57], 0.3, |point| { result.push(point.to_owned()); None });
    assert_eq!(vec![vec![0, 1]], result);

    let quadratic_form = [vec![2., 1.], vec![1., 2.]];
    let mut result = Vec::new();
    fincke_pohst(&ZZ, &RR, Submatrix::from_2d(&quadratic_form), &[4., 7.], 2.1, |point| { result.push(point.to_owned()); None });
    assert_eq!(vec![vec![4, 7], vec![4, 8], vec![4, 6], vec![5, 6], vec![5, 7], vec![3, 8], vec![3, 7]], result);
}

#[test]
fn test_fincke_pohst_3d() {
    let ZZ = StaticRing::<i64>::RING;
    let RR = Real64::RING;

    let quadratic_form = [vec![2., 1., 1.], vec![1., 2., 1.], vec![1., 1., 2.]];
    let mut result = Vec::new();
    fincke_pohst(&ZZ, &RR, Submatrix::from_2d(&quadratic_form), &[0., 0., 0.], 2.001, |point| { result.push(point.to_owned()); None });
    assert_eq!(13, result.len());
    for p in &result {
        assert!(ZZ.pow(p[0] + p[1] + p[2], 2) + ZZ.pow(p[0], 2) + ZZ.pow(p[1], 2) + ZZ.pow(p[2], 2) <= 2);
    }
    
    let quadratic_form = [vec![2., 1., 1.], vec![1., 2., 1.], vec![1., 1., 2.]];
    let mut result = Vec::new();
    fincke_pohst(&ZZ, &RR, Submatrix::from_2d(&quadratic_form), &[0.3, -2.1, 0.7], 0.5, |point| { result.push(point.to_owned()); None });
    assert_eq!(vec![vec![0, -2, 1]], result);
}