use crate::mempool::*;

use super::SwappableVectorViewMut;

///
/// Computes `values_new[i] = values[perm(i)]`.
/// 
pub fn permute<V, T, F, M: MemoryProvider<bool>>(mut values: V, perm: F, memory_provider: &M)
    where V: SwappableVectorViewMut<T>, F: Fn(usize) -> usize
{
    let mut swapped_indices = memory_provider.get_new_init(values.len(), |_| false);
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
pub fn permute_inv<V, T, F, M: MemoryProvider<bool>>(mut values: V, perm: F, memory_provider: &M)
    where V: SwappableVectorViewMut<T>, F: Fn(usize) -> usize
{
    let mut swapped_indices = memory_provider.get_new_init(values.len(), |_| false);
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
    permute(&mut values, |i| permutation[i], &DEFAULT_MEMORY_PROVIDER);
    assert_eq!(values, permutation);
}

#[test]
fn test_permute_inv() {
    let mut values = [2, 1, 7, 5, 6, 3, 4, 0];
    let permutation = [2, 1, 7, 5, 6, 3, 4, 0];
    permute_inv(&mut values, |i| permutation[i], &DEFAULT_MEMORY_PROVIDER);
    assert_eq!(values, [0, 1, 2, 3, 4, 5, 6, 7]);
}
