use std::alloc::{Allocator, Global};

use super::SwappableVectorViewMut;

///
/// Computes `values_new[i] = values[perm(i)]`.
/// 
#[stability::unstable(feature = "enable")]
pub fn permute<V, T, F>(values: V, perm: F)
    where V: SwappableVectorViewMut<T>, F: Fn(usize) -> usize
{
    permute_using_allocator(values, perm, Global)
}

///
/// Computes `values_new[i] = values[perm(i)]`.
/// 
#[stability::unstable(feature = "enable")]
pub fn permute_using_allocator<V, T, F, A: Allocator>(mut values: V, perm: F, allocator: A)
    where V: SwappableVectorViewMut<T>, F: Fn(usize) -> usize
{
    let mut swapped_indices = Vec::with_capacity_in(values.len(), allocator);
    swapped_indices.extend((0..values.len()).map(|_| false));
    let mut start = 0;
    while start < values.len() {
        let mut current = start;
        let mut next = perm(current);
        while !swapped_indices[next] {
            swapped_indices[current] = true;
            values.swap(current, next);
            current = next;
            next = perm(current);
        }
        swapped_indices[current] = true;
        start += 1;
    }
}

///
/// Computes `values_new[perm(i)] = values[i]`.
/// This is the inverse operation to [`permute()`].
/// 
#[stability::unstable(feature = "enable")]
pub fn permute_inv<V, T, F>(values: V, perm: F)
    where V: SwappableVectorViewMut<T>, F: Fn(usize) -> usize
{
    permute_inv_using_allocator(values, perm, Global)
}

///
/// Computes `values_new[perm(i)] = values[i]`.
/// This is the inverse operation to [`permute()`].
/// 
#[stability::unstable(feature = "enable")]
pub fn permute_inv_using_allocator<V, T, F, A: Allocator>(mut values: V, perm: F, allocator: A)
    where V: SwappableVectorViewMut<T>, F: Fn(usize) -> usize
{
    let mut swapped_indices = Vec::with_capacity_in(values.len(), allocator);
    swapped_indices.extend((0..values.len()).map(|_| false));
    let mut start = 0;
    while start < values.len() {
        let mut current = perm(start);
        swapped_indices[start] = true;
        while !swapped_indices[current] {
            swapped_indices[current] = true;
            values.swap(current, start);
            current = perm(current);
        }
        start += 1;
    }
}

#[test]
fn test_permute() {
    let mut values = [0, 1, 2, 3, 4, 5, 6, 7];
    let permutation = [2, 1, 7, 5, 6, 3, 4, 0];
    permute(&mut values, |i| permutation[i]);
    assert_eq!(values, permutation);
}

#[test]
fn test_permute_inv() {
    let mut values = [2, 1, 7, 5, 6, 3, 4, 0];
    let permutation = [2, 1, 7, 5, 6, 3, 4, 0];
    permute_inv(&mut values, |i| permutation[i]);
    assert_eq!(values, [0, 1, 2, 3, 4, 5, 6, 7]);
}
