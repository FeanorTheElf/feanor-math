use std::{cmp::min, convert::TryInto, iter::FusedIterator};

///
/// Iterator over all combinations of the given set of the given size.
/// See also [`combinations()`].
/// 
#[derive(Debug, Clone)]
pub struct IterCombinations<I, F, T> 
    where I: Iterator + Clone, I::Item: Clone, F: FnMut(&[I::Item]) -> T
{
    base: std::iter::Peekable<std::iter::Enumerate<I>>,
    iterators: Box<[std::iter::Peekable<std::iter::Enumerate<I>>]>,
    done: bool,
    converter: F,
    buffer: Box<[I::Item]>
}

impl<I, F, T> Iterator for IterCombinations<I, F, T> 
    where I: Iterator + Clone, I::Item: Clone, F: FnMut(&[I::Item]) -> T
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        } else if self.iterators.len() == 0 {
            self.done = true;
            return Some((self.converter)(&[]));
        }
        let result = (self.converter)(&self.buffer[..]);
        for i in 0..self.iterators.len() - 1 {
            let (its, next_its) = &mut self.iterators[i..i+2].split_at_mut(1);
            let (it, next_it) = (&mut its[0], &mut next_its[0]);
            let can_forward = {
                let (index, _) = it.peek().unwrap();
                let (next_index, _) = next_it.peek().unwrap();
                index + 1 < *next_index
            };
            if can_forward {
                _ = it.next().unwrap();
                self.buffer[i] = it.peek().unwrap().1.clone();
                return Some(result);
            } else {
                // reset and continue with next iterator
                *it = self.base.clone();
                for _ in 0..i {
                    _ = it.next();
                }
                let (_, x) = it.peek().unwrap();
                self.buffer[i] = x.clone();
            }
        }
        if let Some(last_it) = self.iterators.last_mut() {
            _ = last_it.next();
            if let Some(x) = last_it.peek() {
                *self.buffer.last_mut().unwrap() = x.1.clone();
            } else {
                self.done = true;
            }
        }
        return Some(result);
    }
}

impl<I, F, T> FusedIterator for IterCombinations<I, F, T> 
    where I: Iterator + Clone, I::Item: Clone, F: FnMut(&[I::Item]) -> T {}

///
/// Returns an iterator over all combinations of the elements yielded by the base iterator of size `k`.
/// 
/// Since the items yielded by the iterator are collections of dynamic length,
/// creating them might be quite costly. In many applications, this is not really
/// required, and so this function accepts a second argument - a converter function -
/// that is called on each tuple in the cartesian product and its return value
/// is yielded by the product iterator.
/// 
/// Note that clones of the given base iterator must all have the same iteration order.
/// 
/// # Example
/// 
/// ```rust
/// # use feanor_math::iters::combinations;
/// assert_eq!(
///     vec![(1, 2), (1, 3), (2, 3)],
///     combinations(
///         [1, 2, 3].into_iter(), 
///         2, 
///         |a| (a[0], a[1])
///     ).collect::<Vec<_>>()
/// );
/// ```
/// 
pub fn combinations<I, F, T>(it: I, k: usize, converter: F) -> IterCombinations<I, F, T> 
    where I: Iterator + Clone, I::Item: Clone, F: FnMut(&[I::Item]) -> T
{
    let enumerated_it = it.enumerate().peekable();
    let mut start_iterators = Vec::with_capacity(k);
    let mut buffer = Vec::with_capacity(k);
    let mut start_it = enumerated_it.clone();
    for _ in 0..k {
        start_iterators.push(start_it.clone());
        if start_it.peek().is_none() {
            return IterCombinations {
                base: enumerated_it,
                iterators: start_iterators.into_boxed_slice(),
                done: true,
                converter: converter,
                buffer: buffer.into_boxed_slice()
            };
        }
        let (_, x) = start_it.next().unwrap();
        buffer.push(x);
    }
    return IterCombinations {
        base: enumerated_it,
        iterators: start_iterators.into_boxed_slice(),
        done: false,
        converter: converter,
        buffer: buffer.into_boxed_slice()
    };
}

///
/// Clones every element of the given slice and returns an owned pointer to the copy.
/// 
pub fn clone_slice<T>(slice: &[T]) -> Box<[T]> 
    where T: Clone
{
    let vec: Vec<T> = slice.iter().cloned().collect();
    return vec.into_boxed_slice();
}

///
/// Clones every element of the given array.
/// 
pub fn clone_array<T, const N: usize>(slice: &[T]) -> [T; N] 
    where T: Copy
{
    slice.try_into().unwrap()
}

///
/// Returns an iterator over all size-`k` subsets of the
/// elements returned by the given iterator.
/// 
/// This is a special version of [`combinations()`], which
/// accepts a converter function and thus can avoid cloning
/// each subset.
/// 
pub fn basic_combinations<I>(it: I, k: usize) -> impl Iterator<Item = Box<[I::Item]>>
    where I: Iterator + Clone, I::Item: Clone, 
{
    combinations(it, k, clone_slice)
}

///
/// Calls the given `converter` function once for each subset of the elements
/// returned by the given iterator, and returns an iterator over the
/// outputs of the `converter` function.
/// 
pub fn powerset<I, F, T>(it: I, converter: F) -> impl Iterator<Item = T>
    where I: Iterator + Clone, I::Item: Clone, F: Clone + FnMut(&[I::Item]) -> T
{
    let len = it.clone().count();
    (0..=len).flat_map(move |i| combinations(it.clone(), i, converter.clone()))
}

///
/// Returns an iterator over all subsets of the
/// elements returned by the given iterator.
/// 
/// This is a special version of [`powerset()`], which
/// accepts a converter function and thus can avoid cloning
/// each subset.
/// 
pub fn basic_powerset<I>(it: I) -> impl Iterator<Item = Box<[I::Item]>>
    where I: Iterator + Clone, I::Item: Clone
{
    powerset(it, clone_slice)
}

///
/// Iterator over the multiset combinations of a set and a given size.
/// See also [`multiset_combinations()`].
/// 
#[derive(Debug, Clone)]
pub struct MultisetCombinations<'a, F, T>
    where F: FnMut(&[usize]) -> T
{
    converter: F,
    superset: &'a [usize],
    current: Option<Box<[usize]>>,
    last_moved: usize,
    first_trailing_empty: usize
}

impl<'a, F, T> Iterator for MultisetCombinations<'a, F, T>
    where F: FnMut(&[usize]) -> T
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.current.is_none() {
            return None;
        }
        let current = &mut self.current.as_mut().unwrap();
        let result = (self.converter)(current);
        let mut removed = 0;
        let mut found_empty_place = self.last_moved + 1 < self.first_trailing_empty;
        while !found_empty_place || current[self.last_moved] == 0 {
            found_empty_place |= current[self.last_moved] < self.superset[self.last_moved];
            removed += current[self.last_moved];
            current[self.last_moved] = 0;
            if self.last_moved == 0 {
                self.current = None;
                return Some(result);
            }
            self.last_moved -= 1;
        }
        removed += 1;
        current[self.last_moved] -= 1;
        while removed > 0 {
            self.last_moved += 1;
            if current[self.last_moved] + removed > self.superset[self.last_moved] {
                removed = current[self.last_moved] + removed - self.superset[self.last_moved];
                current[self.last_moved] = self.superset[self.last_moved];
            } else {
                current[self.last_moved] += removed;
                removed = 0;
            }
        }
        return Some(result);
    }
}

impl<'a, F, T> FusedIterator for MultisetCombinations<'a, F, T>
    where F: FnMut(&[usize]) -> T {}

///
/// Returns an iterator over all multi-subsets of a given set with a specified size.
/// 
/// Since the items yielded by the iterator are collections of dynamic length,
/// creating them might be quite costly. In many applications, this is not really
/// required, and so this function accepts a second argument - a converter function -
/// that is called on each tuple in the cartesian product and its return value
/// is yielded by the product iterator.
/// 
/// Note that clones of the given base iterator must all have the same iteration order.
/// 
/// This iterator returns its elements in descending lexicographic order, so the first
/// yielded element is the largest one w.r.t. lexicographic order.
/// 
/// # Example
/// ```rust
/// # use feanor_math::iters::multiset_combinations;
/// assert_eq!(
///     vec![(1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
///     multiset_combinations(
///         &[1, 2, 3], 
///         2, 
///         |a| (a[0], a[1], a[2])
///     ).collect::<Vec<_>>()
/// );
/// ```
/// ```rust
/// # use feanor_math::iters::multiset_combinations;
/// assert_eq!(
///     vec![(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
///     multiset_combinations(
///         &[2, 2, 2], 
///         2, 
///         |a| (a[0], a[1], a[2])
///     ).collect::<Vec<_>>()
/// );
/// ```
/// 
pub fn multiset_combinations<'a, F, T>(multiset: &'a [usize], size: usize, converter: F) -> MultisetCombinations<'a, F, T>
    where F: FnMut(&[usize]) -> T
{
    assert!(multiset.len() > 0);
    if size > multiset.iter().copied().sum::<usize>() {
        return MultisetCombinations {
            converter: converter,
            superset: multiset,
            first_trailing_empty: 0,
            current: None,
            last_moved: 0
        };
    }
    let mut start = (0..multiset.len()).map(|_| 0).collect::<Vec<_>>().into_boxed_slice();
    let mut to_insert = size;
    let mut last_moved = 0;
    let mut i = 0;
    while to_insert > 0 {
        start[i] = min(multiset[i], to_insert);
        last_moved = i;
        to_insert -= start[i];
        i += 1;
    }
    let mut trailing_empty_entries = 0;
    while trailing_empty_entries < multiset.len() && multiset[multiset.len() - trailing_empty_entries - 1] == 0 {
        trailing_empty_entries += 1;
    }
    return MultisetCombinations {
        converter: converter,
        superset: multiset,
        first_trailing_empty: multiset.len() - trailing_empty_entries,
        current: Some(start),
        last_moved: last_moved
    };
}

///
/// Iterator over the elements of the cartesian product of multiple iterators.
/// See also [`multi_cartesian_product()`].
/// 
#[derive(Debug)]
pub struct MultiProduct<I, F, G, T> 
    where I: Iterator + Clone, 
        F: FnMut(&[I::Item]) -> T,
        G: Clone + Fn(usize, &I::Item) -> I::Item
{
    base_iters: Vec<I>,
    current_iters: Vec<I>,
    current: Vec<I::Item>,
    clone_el: G,
    converter: F,
    done: bool
}

impl<I, F, G, T> Clone for MultiProduct<I, F, G, T>
    where I: Iterator + Clone, 
        F: Clone + FnMut(&[I::Item]) -> T,
        G: Clone + Fn(usize, &I::Item) -> I::Item
{
    fn clone(&self) -> Self {
        MultiProduct {
            base_iters: self.base_iters.clone(),
            current_iters: self.current_iters.clone(),
            current: self.current.iter().enumerate().map(|(i, x)| (self.clone_el)(i, x)).collect(),
            clone_el: self.clone_el.clone(),
            converter: self.converter.clone(),
            done: self.done
        }
    }
}

impl<I, F, G, T> Iterator for MultiProduct<I, F, G, T>
    where I: Iterator + Clone, 
        F: FnMut(&[I::Item]) -> T,
        G: Clone + Fn(usize, &I::Item) -> I::Item
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let result = (self.converter)(&self.current[..]);
        let mut i = self.base_iters.len();
        self.done = true;
        while i > 0 {
            i = i - 1;
            if let Some(val) = self.current_iters[i].next() {
                self.current[i] = val;
                self.done = false;
                for j in (i + 1)..self.base_iters.len() {
                    self.current_iters[j] = self.base_iters[j].clone();
                    self.current[j] = self.current_iters[j].next().unwrap();
                }
                break;
            }
        }
        return Some(result);
    }
}

impl<I, F, G, T> FusedIterator for MultiProduct<I, F, G, T>
    where I: Iterator + Clone, 
        F: Clone + FnMut(&[I::Item]) -> T,
        G: Clone + Fn(usize, &I::Item) -> I::Item
{}

///
/// Creates an iterator that computes the cartesian product of the elements
/// yielded by a number of iterators. These iterators are given by one iterator
/// over them.
/// 
/// Since the items yielded by the iterator are collections of dynamic length,
/// creating them might be quite costly. In many applications, this is not really
/// required, and so this function accepts a second argument - a converter function -
/// that is called on each tuple in the cartesian product and its return value
/// is yielded by the product iterator.
/// 
/// Note that clones of the given base iterator must all have the same iteration order.
/// 
/// # Example
/// ```rust
/// # use feanor_math::iters::multi_cartesian_product;
/// assert_eq!(
///     vec![(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)],
///     multi_cartesian_product(
///         vec![(2..=3), (0..=2)].into_iter(),
///         |x| (x[0], x[1]),
///         |_, x| *x
///     ).collect::<Vec<_>>()
/// );
/// ```
/// 
pub fn multi_cartesian_product<J, F, G, T>(iters: J, converter: F, clone_el: G) -> MultiProduct<J::Item, F, G, T>
    where J: Iterator, 
        J::Item: Iterator + Clone, 
        F: FnMut(&[<J::Item as Iterator>::Item]) -> T,
        G: Clone + Fn(usize, &<J::Item as Iterator>::Item) -> <J::Item as Iterator>::Item
{
    let base_iters = iters.collect::<Vec<_>>();
    let mut current_iters = base_iters.clone();
    let mut current = Vec::with_capacity(current_iters.len());
    for it in current_iters.iter_mut() {
        if let Some(v) = it.next() {
            current.push(v);
        } else {
            return MultiProduct {
                done: true,
                converter: converter,
                base_iters: base_iters,
                clone_el: clone_el,
                current_iters: current_iters,
                current: current
            };
        }
    }
    return MultiProduct {
        done: false,
        converter: converter,
        base_iters: base_iters,
        current_iters: current_iters,
        clone_el: clone_el,
        current: current
    };
}

///
/// An iterator that "condenses" a given iterator, i.e. combines a variable number
/// of base iterator elements into a single element.
/// 
/// Use through the function [`condense()`].
/// 
#[derive(Debug)]
pub struct CondenseIter<I, F, T>
    where I: Iterator, F: FnMut(I::Item) -> Option<T>
{
    base_iter: I,
    f: F
}

impl<I, F, T> Iterator for CondenseIter<I, F, T>
    where I: Iterator, F: FnMut(I::Item) -> Option<T>
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        while let Some(el) = self.base_iter.next() {
            if let Some(result) = (self.f)(el) {
                return Some(result);
            }
        }
        return None;
    }
}

impl<I, F, T> FusedIterator for CondenseIter<I, F, T>
    where I: FusedIterator, F: FnMut(I::Item) -> Option<T>
{}

///
/// Creates an iterator that "condenses" the given iterator.
/// Concretely, the new iterator processes a variable amount
/// of input items to produce an output item.
/// 
/// This processing is done using the passed function. It is
/// called repeatedly on input iterator items, until it returns
/// some value, which is the next value yielded by the output
/// iterator.
/// 
/// # Example
/// ```rust
/// # use feanor_math::iters::condense;
/// let mut accumulator = 0;
/// assert_eq!(vec![6, 9, 6, 7], condense(vec![1, 2, 3, 4, 5, 6, 7].into_iter(), move |a| {
///     accumulator += a;
///     if accumulator >= 5 {
///         let result = accumulator;
///         accumulator = 0;
///         return Some(result);
///     } else {
///         return None;
///     }
/// }).collect::<Vec<i64>>());
/// ```
/// 
pub fn condense<I, F, T>(iter: I, f: F) -> CondenseIter<I, F, T>
    where I: Iterator, F: FnMut(I::Item) -> Option<T>
{
    CondenseIter { base_iter: iter, f: f }
}

#[test]
fn test_converted_combinations() {
    let a = [2, 3, 5, 7];
    assert_eq!(1, combinations(a.iter(), 0, |_| 0).count());
    assert_eq!(4, combinations(a.iter(), 1, |_| 0).count());
    assert_eq!(6, combinations(a.iter(), 2, |_| 0).count());
    assert_eq!(1, combinations(a.iter(), 4, |_| 0).count());
    assert_eq!(0, combinations(a.iter(), 5, |_| 0).count());
}

#[test]
fn test_powerset() {
    let a = [1, 2, 3, 4];
    assert_eq!(16, basic_powerset(a.iter()).count());

    let a = [2, 3];
    assert_eq!(vec![
        vec![].into_boxed_slice(),
        vec![2].into_boxed_slice(),
        vec![3].into_boxed_slice(),
        vec![2, 3].into_boxed_slice() 
    ], basic_powerset(a.iter().copied()).collect::<Vec<_>>());

    let a = [1, 2, 3];
    assert_eq!(vec![
        vec![].into_boxed_slice(),
        vec![1].into_boxed_slice(),
        vec![2].into_boxed_slice(),
        vec![3].into_boxed_slice(),
        vec![1, 2].into_boxed_slice(),
        vec![1, 3].into_boxed_slice(),
        vec![2, 3].into_boxed_slice(),
        vec![1, 2, 3].into_boxed_slice()
    ], basic_powerset(a.iter().copied()).collect::<Vec<_>>());
}

#[allow(deprecated)]
#[test]
fn test_multi_cartesian_product() {
    let a = [0, 1];
    let b = [0, 1];
    let c = [-1, 1];
    let all = [a, b, c];
    let it = multi_cartesian_product(
        all.iter().map(|l| l.iter().map(|x| *x)), 
        |x| [x[0], x[1], x[2]],
        |_, x| *x
    );
    let expected = vec![
        [0, 0, -1],
        [0, 0, 1],
        [0, 1, -1],
        [0, 1, 1],
        [1, 0, -1],
        [1, 0, 1],
        [1, 1, -1],
        [1, 1, 1]
    ];
    assert_eq!(expected, it.collect::<Vec<_>>());
}

#[allow(trivial_casts)]
#[test]
fn test_multiset_combinations() {
    let a = [1, 2, 3, 1];
    let mut iter = multiset_combinations(&a, 3, clone_slice);
    assert_eq!(&[1, 2, 0, 0][..], &*iter.next().unwrap());
    assert_eq!(&[1, 1, 1, 0][..], &*iter.next().unwrap());
    assert_eq!(&[1, 1, 0, 1][..], &*iter.next().unwrap());
    assert_eq!(&[1, 0, 2, 0][..], &*iter.next().unwrap());
    assert_eq!(&[1, 0, 1, 1][..], &*iter.next().unwrap());

    assert_eq!(&[0, 2, 1, 0][..], &*iter.next().unwrap());
    assert_eq!(&[0, 2, 0, 1][..], &*iter.next().unwrap());
    assert_eq!(&[0, 1, 2, 0][..], &*iter.next().unwrap());
    assert_eq!(&[0, 1, 1, 1][..], &*iter.next().unwrap());

    assert_eq!(&[0, 0, 3, 0][..], &*iter.next().unwrap());
    assert_eq!(&[0, 0, 2, 1][..], &*iter.next().unwrap());
    assert_eq!(None, iter.next());

    assert_eq!(&([] as [[usize; 1]; 0]), &multiset_combinations(&[0], 1, clone_array::<usize, 1>).collect::<Vec<_>>()[..]);
    assert_eq!(&([[0]] as [[usize; 1]; 1]), &multiset_combinations(&[0], 0, clone_array::<usize, 1>).collect::<Vec<_>>()[..]);

    let a = [0, 2, 0, 1];
    let mut iter = multiset_combinations(&a, 2, clone_slice);
    assert_eq!(&[0, 2, 0, 0][..], &*iter.next().unwrap());
    assert_eq!(&[0, 1, 0, 1][..], &*iter.next().unwrap());
    assert_eq!(None, iter.next());

    let a = [0, 3, 0, 0, 2, 0];
    let mut iter = multiset_combinations(&a, 3, clone_slice);
    assert_eq!(&[0, 3, 0, 0, 0, 0][..], &*iter.next().unwrap());
    assert_eq!(&[0, 2, 0, 0, 1, 0][..], &*iter.next().unwrap());
    assert_eq!(&[0, 1, 0, 0, 2, 0][..], &*iter.next().unwrap());
    assert_eq!(None, iter.next());

    let a = [0, 3, 0, 0, 2, 2, 0];
    let mut iter = multiset_combinations(&a, 3, clone_slice);
    assert_eq!(&[0, 3, 0, 0, 0, 0, 0][..], &*iter.next().unwrap());
    assert_eq!(&[0, 2, 0, 0, 1, 0, 0][..], &*iter.next().unwrap());
    assert_eq!(&[0, 2, 0, 0, 0, 1, 0][..], &*iter.next().unwrap());
    assert_eq!(&[0, 1, 0, 0, 2, 0, 0][..], &*iter.next().unwrap());
    assert_eq!(&[0, 1, 0, 0, 1, 1, 0][..], &*iter.next().unwrap());
    assert_eq!(&[0, 1, 0, 0, 0, 2, 0][..], &*iter.next().unwrap());
    assert_eq!(&[0, 0, 0, 0, 2, 1, 0][..], &*iter.next().unwrap());
    assert_eq!(&[0, 0, 0, 0, 1, 2, 0][..], &*iter.next().unwrap());
    assert_eq!(None, iter.next());

    let a = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0];
    let mut iter = multiset_combinations(&a, 10, clone_slice);
    assert_eq!(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0][..], &*iter.next().unwrap());
    assert_eq!(1000, iter.count());
}

#[test]
fn test_multiset_combinations_k_unlimited() {
    fn fac(n: usize) -> usize {
        if n == 0 { 1 } else { n * fac(n - 1) }
    }
    let a = [10, 10, 10, 10, 10, 10];
    assert_eq!(1, multiset_combinations(&a[..], 0, |_| ()).count());
    assert_eq!(6, multiset_combinations(&a[..], 1, |_| ()).count());
    assert_eq!(fac(6 + 8 - 1) / fac(6 - 1) / fac(8), multiset_combinations(&a[..], 8, |_| ()).count());
}