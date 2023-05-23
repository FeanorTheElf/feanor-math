use crate::ring::*;
use crate::integer::*;

pub fn generic_abs_square_and_multiply<T, U, F, H, I>(base: U, power: &El<I>, int_ring: I, mut square: F, mut multiply_base: H, identity: T) -> T
    where I: IntegerRingStore,
        I::Type: IntegerRing,
        F: FnMut(T) -> T, H: FnMut(&U, T) -> T
{
    if int_ring.is_zero(&power) {
        return identity;
    } else if int_ring.is_one(&power) {
        return multiply_base(&base, identity);
    }

    let mut result = identity;
    for i in (0..=int_ring.abs_highest_set_bit(power).unwrap()).rev() {
        if int_ring.abs_is_bit_set(power, i) {
            result = multiply_base(&base, square(result));
        } else {
            result = square(result);
        }
    }
    return result;
}

pub fn generic_pow<R: ?Sized, S: ?Sized, I>(base: El<R>, power: &El<I>, base_ring: &R, ring: &S, int_ring: &I) -> El<S>
    where R: RingStore, 
        S: RingStore,
        S::Type: CanonicalHom<R::Type>,
        I: IntegerRingStore,
        I::Type: IntegerRing
{
    let hom = ring.get_ring().has_canonical_hom(base_ring.get_ring()).unwrap();
    generic_abs_square_and_multiply(
        base, 
        power, 
        int_ring, 
        |mut x| {
            ring.square(&mut x);
            x
        }, 
        |x, mut y| { 
            ring.get_ring().mul_assign_map_in_ref(base_ring.get_ring(), &mut y, x, &hom);
            y
        }, 
        ring.one()
    )
}

#[cfg(test)]
use crate::primitive_int::*;

#[test]
fn test_pow() {
    assert_eq!(3 * 3, generic_abs_square_and_multiply(3, &2, StaticRing::<i64>::RING, |a| a * a, |a, b| *a * b, 1));
    assert_eq!(3 * 3 * 3 * 3 * 3, generic_abs_square_and_multiply(3, &5, StaticRing::<i64>::RING, |a| a * a, |a, b| *a * b, 1));
}
