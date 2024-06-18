use std::{marker::PhantomData, num::NonZero};

use super::*;

pub struct VectorFnIter<V: VectorFn<T>, T> {
    content: V,
    begin: usize,
    end: usize,
    element: PhantomData<T>
}

impl<V: VectorFn<T>, T> VectorFnIter<V, T> {

    pub fn new(content: V) -> Self {
        Self {
            end: content.len(),
            content: content,
            begin: 0,
            element: PhantomData
        }
    }
}

impl<V: Clone + VectorFn<T>, T> Clone for VectorFnIter<V, T> {
    
    fn clone(&self) -> Self {
        Self {
            begin: self.begin,
            end: self.end,
            content: self.content.clone(),
            element: PhantomData
        }
    }
}

impl<V: Copy + VectorFn<T>, T> Copy for VectorFnIter<V, T> {}

impl<V: VectorFn<T>, T> Iterator for VectorFnIter<V, T> {

    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.begin < self.end {
            self.begin += 1;
            return Some(self.content.at(self.begin - 1));
        } else {
            return None;
        }
    }

    fn advance_by(&mut self, n: usize) -> Result<(), std::num::NonZero<usize>> {
        if self.begin + n <= self.end {
            self.begin += n;
            return Ok(());
        } else {
            let lacking_elements = n - (self.end - self.begin);
            self.begin = self.end;
            return Err(NonZero::new(lacking_elements).unwrap())
        }
    }
}

impl<V: VectorFn<T>, T> DoubleEndedIterator for VectorFnIter<V, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.begin < self.end {
            self.end -= 1;
            return Some(self.content.at(self.end));
        } else {
            return None;
        }
    }

    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        if self.begin + n <= self.end {
            self.end -= n;
            return Ok(());
        } else {
            let lacking_elements = n - (self.end - self.begin);
            self.end = self.begin;
            return Err(NonZero::new(lacking_elements).unwrap())
        }
    }
}

impl<V: VectorFn<T>, T> ExactSizeIterator for VectorFnIter<V, T> {
    fn len(&self) -> usize {
        self.end - self.begin
    }
}

pub struct CloneElFn<V: VectorView<T>, T, F: Fn(&T) -> T> {
    content: V,
    clone_el: F,
    element: PhantomData<T>
}

impl<V: Clone + VectorView<T>, T, F: Clone + Fn(&T) -> T> Clone for CloneElFn<V, T, F> {
    
    fn clone(&self) -> Self {
        Self {
            clone_el: self.clone_el.clone(),
            content: self.content.clone(),
            element: PhantomData
        }
    }
}

impl<V: Copy + VectorView<T>, T, F: Copy + Fn(&T) -> T> Copy for CloneElFn<V, T, F> {}

impl<V: VectorView<T>, T, F: Fn(&T) -> T> CloneElFn<V, T, F> {

    pub fn new(content: V, clone_el: F) -> Self {
        Self {
            content: content,
            clone_el: clone_el,
            element: PhantomData
        }
    }
}

impl<V: VectorView<T>, T, F: Fn(&T) -> T> VectorFn<T> for CloneElFn<V, T, F> {
    
    fn at(&self, i: usize) -> T {
        (self.clone_el)(self.content.at(i))
    }

    fn len(&self) -> usize {
        self.content.len()
    }
}

pub struct VectorViewFn<'a, V: ?Sized + VectorView<T>, T: ?Sized> {
    content: &'a V,
    element: PhantomData<T>
}

impl<'a, V: ?Sized + VectorView<T>, T: ?Sized> Clone for VectorViewFn<'a, V, T> {
    
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V: ?Sized + VectorView<T>, T: ?Sized> Copy for VectorViewFn<'a, V, T> {}

impl<'a, V: ?Sized + VectorView<T>, T: ?Sized> VectorViewFn<'a, V, T> {
    pub fn new(content: &'a V) -> Self {
        Self {
            content: content,
            element: PhantomData
        }
    }
}

impl<'a, V: ?Sized + VectorView<T>, T: ?Sized> VectorFn<&'a T> for VectorViewFn<'a, V, T> {
    
    fn at(&self, i: usize) -> &'a T {
        self.content.at(i)
    }

    fn len(&self) -> usize {
        self.content.len()
    }
}