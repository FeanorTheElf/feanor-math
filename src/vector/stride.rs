use std::marker::PhantomData;

use crate::vector::*;

pub struct Stride<T, V> 
    where V: VectorView<T>
{
    base_view: V,
    base_element: PhantomData<T>,
    stride: usize
}

impl<T, V> Stride<T, V> 
    where V: VectorView<T>
{
    pub const fn new(base_view: V, stride: usize) -> Self {
        Stride {
            base_view: base_view,
            stride: stride,
            base_element: PhantomData
        }
    }
}

impl<T, V> Clone for Stride<T, V> 
    where V: VectorView<T> + Clone
{
    fn clone(&self) -> Self {
        Stride::new(self.base_view.clone(), self.stride)
    }
}

impl<T, V> Copy for Stride<T, V> 
    where V: VectorView<T> + Copy
{}

impl<T, V> VectorView<T> for Stride<T, V> 
    where V: VectorView<T>
{
    fn at(&self, i: usize) -> &T {
        self.base_view.at(i * self.stride)
    }

    fn len(&self) -> usize {
        if self.base_view.len() == 0 {
            0
        } else {
            (self.base_view.len() - 1) / self.stride + 1
        }
    }
}

impl<T, V> VectorViewMut<T> for Stride<T, V> 
    where V: VectorViewMut<T>
{
    fn at_mut(&mut self, i: usize) -> &mut T {
        self.base_view.at_mut(i * self.stride)
    }
}

impl<T, V> SwappableVectorViewMut<T> for Stride<T, V> 
    where V: SwappableVectorViewMut<T>
{
    fn swap(&mut self, i: usize, j: usize) {
        self.base_view.swap(i * self.stride, j * self.stride)
    }
}

#[test]
fn test_stride() {
    let vec = [0, 1, 2, 3, 4, 5, 6, 7];
    let zero: [i32; 0] = [];
    assert_eq!(0, zero.stride(1).len());
    assert_eq!(4, vec.stride(2).len());
    assert_eq!(3, vec.stride(3).len());
    assert_eq!(6, *vec.stride(2).at(3));
    assert_eq!(0, *vec.stride(3).at(0));
    assert_eq!(3, *vec.stride(3).at(1));
}