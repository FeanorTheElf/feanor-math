use std::collections::HashMap;
use std::collections::hash_map;
use std::collections::hash_map::Entry;

use crate::ring::*;
use crate::seq::*;

pub struct SparseMapVector<R: RingStore> {
    data: HashMap<usize, El<R>>,
    modify_entry: (usize, El<R>),
    zero: El<R>,
    ring: R,
    len: usize
}

impl<R: RingStore> SparseMapVector<R> {

    pub fn new(len: usize, ring: R) -> Self {
        SparseMapVector {
            data: HashMap::new(), 
            modify_entry: (usize::MAX, ring.zero()),
            zero: ring.zero(),
            ring: ring,
            len: len
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn set_len(&mut self, new_len: usize) {
        if new_len < self.len() {
            for (i, _) in self.nontrivial_entries() {
                debug_assert!(i < new_len);
            }
        }
        self.len = new_len;
    }

    #[stability::unstable(feature = "enable")]
    pub fn scan<F>(&mut self, mut f: F)
        where F: FnMut(usize, &mut El<R>)
    {
        self.enter_in_map((usize::MAX, self.ring.zero()));
        self.data.retain(|i, c| {
            f(*i, c);
            !self.ring.is_zero(c)
        });
    }

    #[cfg(test)]
    fn check_consistency(&self) {
        assert!(self.ring.is_zero(&self.modify_entry.1) || self.modify_entry.0 < self.len());
    }

    fn enter_in_map(&mut self, new_modify_entry: (usize, El<R>)) {
        if self.modify_entry.0 != usize::MAX {
            let (index, value) = std::mem::replace(&mut self.modify_entry, new_modify_entry);
            match self.data.entry(index) {
                Entry::Occupied(mut e) if !self.ring.is_zero(&value) => { *e.get_mut() = value; },
                Entry::Occupied(e) => { _ = e.remove(); },
                Entry::Vacant(e) if !self.ring.is_zero(&value) => { _ = e.insert(value); },
                Entry::Vacant(_) => {}
            };
        } else {
            self.modify_entry = new_modify_entry;
        }
    }
}

impl<R: RingStore + Clone> Debug for SparseMapVector<R> {
    
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut output = f.debug_map();
        for (key, value) in self.nontrivial_entries() {
            _ = output.entry(&key, &self.ring.formatted_el(value));
        }
        output.finish()
    }
}

impl<R: RingStore + Clone> Clone for SparseMapVector<R> {

    fn clone(&self) -> Self {
        SparseMapVector { 
            data: self.data.iter().map(|(i, c)| (*i, self.ring.clone_el(c))).collect(), 
            modify_entry: (self.modify_entry.0, self.ring.clone_el(&self.modify_entry.1)), 
            zero: self.ring.clone_el(&self.zero), 
            ring: self.ring.clone(), 
            len: self.len
        }
    }
}

impl<R: RingStore> VectorView<El<R>> for SparseMapVector<R> {

    fn at(&self, i: usize) -> &El<R> {
        assert!(i < self.len());
        if i == self.modify_entry.0 {
            &self.modify_entry.1
        } else if let Some(res) = self.data.get(&i) {
            res
        } else {
            &self.zero
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn specialize_sparse<Op: SparseVectorViewOperation<El<R>, Self>>(op: Op) -> Op::Output {
        op.execute()
    }
}

pub struct SparseMapVectorIter<'a, R>
    where R: RingStore
{
    base: hash_map::Iter<'a, usize, El<R>>,
    skip: usize,
    once: Option<&'a El<R>>
}

impl<'a, R> Iterator for SparseMapVectorIter<'a, R>
    where R: RingStore
{
    type Item = (usize, &'a El<R>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(start) = self.once {
            self.once = None;
            return Some((self.skip, start));
        } else {
            while let Some((index, element)) = self.base.next() {
                if *index != self.skip {
                    return Some((*index, element));
                }
            }
            return None;
        }
    }
}

impl<R: RingStore> VectorViewSparse<El<R>> for SparseMapVector<R> {

    type Iter<'a> = SparseMapVectorIter<'a, R>
        where Self: 'a;

    fn nontrivial_entries<'a>(&'a self) -> Self::Iter<'a> {
        SparseMapVectorIter {
            base: self.data.iter(),
            skip: self.modify_entry.0,
            once: if !self.ring.is_zero(&self.modify_entry.1) { Some(&self.modify_entry.1) } else { None }
        }
    }
}

impl<R: RingStore> VectorViewMut<El<R>> for SparseMapVector<R> {

    fn at_mut(&mut self, i: usize) -> &mut El<R> {
        assert!(i < self.len());
        if i == self.modify_entry.0 {
            return &mut self.modify_entry.1;
        }
        let new_value = self.ring.clone_el(self.data.get(&i).unwrap_or(&self.zero));
        self.enter_in_map((i, new_value));
        return &mut self.modify_entry.1;
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;

#[cfg(test)]
fn assert_vector_eq<const N: usize>(vec: &SparseMapVector<StaticRing<i64>>, values: [i64; N]) {
    assert_eq!(vec.len(), N);
    vec.check_consistency();
    for i in 0..N {
        // at_mut() might change the vector, so don't test that
        assert_eq!(*vec.at(i), values[i]);
    }
}

#[test]
fn test_at_mut() {
    LogAlgorithmSubscriber::init_test();
    let ring = StaticRing::<i64>::RING;
    let mut vector = SparseMapVector::new(5, ring);

    assert_vector_eq(&mut vector, [0, 0, 0, 0, 0]);
    let mut entry = vector.at_mut(1);
    assert_eq!(0, *entry);
    *entry = 3;
    assert_vector_eq(&mut vector, [0, 3, 0, 0, 0]);

    entry = vector.at_mut(4);
    assert_eq!(0, *entry);
    *entry = -1;
    assert_vector_eq(&mut vector, [0, 3, 0, 0, -1]);
    
    entry = vector.at_mut(1);
    assert_eq!(3, *entry);
    *entry = 4;
    assert_vector_eq(&mut vector, [0, 4, 0, 0, -1]);

    entry = vector.at_mut(1);
    assert_eq!(4, *entry);
    *entry = 5;
    assert_vector_eq(&mut vector, [0, 5, 0, 0, -1]);

    entry = vector.at_mut(3);
    assert_eq!(0, *entry);
    *entry = 0;
    assert_vector_eq(&mut vector, [0, 5, 0, 0, -1]);
}

#[test]
fn test_nontrivial_entries() {
    LogAlgorithmSubscriber::init_test();
    let ring = StaticRing::<i64>::RING;
    let mut vector = SparseMapVector::new(5, ring);
    assert_eq!(vector.nontrivial_entries().collect::<HashMap<_, _>>(), [].into_iter().collect());
    *vector.at_mut(1) = 3;
    assert_eq!(vector.nontrivial_entries().collect::<HashMap<_, _>>(), [(1, &3)].into_iter().collect());
    *vector.at_mut(4) = -1;
    assert_eq!(vector.nontrivial_entries().collect::<HashMap<_, _>>(), [(1, &3), (4, &-1)].into_iter().collect());

    *vector.at_mut(1) = 4;
    assert_eq!(vector.nontrivial_entries().collect::<HashMap<_, _>>(), [(1, &4), (4, &-1)].into_iter().collect());
    *vector.at_mut(1) = 0;
    assert_eq!(vector.nontrivial_entries().collect::<HashMap<_, _>>(), [(4, &-1)].into_iter().collect());
    *vector.at_mut(1) = 5;
    assert_eq!(vector.nontrivial_entries().collect::<HashMap<_, _>>(), [(1, &5), (4, &-1)].into_iter().collect());

    *vector.at_mut(3) = 0;
    assert_eq!(vector.nontrivial_entries().collect::<HashMap<_, _>>(), [(1, &5), (4, &-1)].into_iter().collect());
    *vector.at_mut(4) = -2;
    assert_eq!(vector.nontrivial_entries().collect::<HashMap<_, _>>(), [(1, &5), (4, &-2)].into_iter().collect());

    *vector.at_mut(1) = 0;
    assert_eq!(vector.nontrivial_entries().count(), 1);
    *vector.at_mut(4) = 0;
    assert_eq!(vector.nontrivial_entries().count(), 0);
}

#[test]
fn test_scan() {
    LogAlgorithmSubscriber::init_test();
    let ring = StaticRing::<i64>::RING;
    let mut vector = SparseMapVector::new(5, ring);
    *vector.at_mut(1) = 2;
    *vector.at_mut(3) = 1;
    *vector.at_mut(4) = 0;
    vector.scan(|_, c| {
        *c -= 1;
    });
    assert_vector_eq(&vector, [0, 1, 0, 0, 0]);
    assert_eq!(vector.nontrivial_entries().collect::<HashMap<_, _>>(), [(1, &1)].into_iter().collect());
}