
pub trait VectorView<T> {

    type Subvector: VectorView<T>;

    fn len(&self) -> usize;
    fn at(&self, index: usize) -> &T;
    fn create_subvector(self, from: usize, to: usize) -> Self::Subvector;

    fn assert_in_range(&self, index: usize) {
        assert!(index < self.len(), "Vector index {} out of range 0..{}", index, self.len());
    }
}

pub trait VectorViewMut<T>: VectorView<T> {

    type SubvectorMut: VectorViewMut<T>;

    fn at_mut(&mut self, index: usize) -> &mut T;
    fn swap(&mut self, i: usize, j: usize);

    ///
    /// Currently, there is no way to strengthen the associated type bound in
    /// a subtrait, i.e. require `Self::Subvector: VectorViewMut` in VectorViewMut.
    /// "Faking" this by adding a `where Self::Subvector: VectorViewMut` to the
    /// subtrait definition does not work:
    /// Adding such a where-clause then requires adding the corresponding where-clause
    /// to every use of VectorViewMut, and besides the inconvenience, this causes a
    /// recursion. In other words, we would have to add
    /// ```text
    ///     where V: VectorViewMut<T>,
    ///           V::Subvector: VectorViewMut<T>,
    ///           V::Subvector::Subvector: VectorViewMut<T>,
    ///           ...
    /// ```
    /// This is the best workaround I found. Just implement this function as the identity...
    /// 
    fn cast_subvector(subvector: Self::Subvector) -> Self::SubvectorMut;
}

pub struct SubvectorView<V> {
    data: V,
    from: usize,
    to: usize
}

impl<V> Clone for SubvectorView<V>
    where V: Clone
{
    fn clone(&self) -> Self {
        SubvectorView { data: self.data.clone(), from: self.from, to: self.to }
    }
}

impl<V> Copy for SubvectorView<V>
    where V: Copy
{}

impl<V> SubvectorView<V> {

    pub const fn new(data: V, from: usize, to: usize) -> Self {
        SubvectorView { data: data, from: from, to: to }
    }
}

impl<T, V> VectorView<T> for SubvectorView<V>
    where V: VectorView<T>
{
    type Subvector = Self;

    fn len(&self) -> usize {
        self.to - self.from
    }

    fn at(&self, index: usize) -> &T {
        self.assert_in_range(index);
        self.data.at(index + self.from)
    }

    fn create_subvector(self, from: usize, to: usize) -> Self::Subvector {
        self.assert_in_range(from);
        assert!(to <= self.len());
        assert!(from <= to);
        SubvectorView::new(self.data, self.from + from, self.to + to)
    }
}

impl<T, V> VectorViewMut<T> for SubvectorView<V>
    where V: VectorViewMut<T>
{
    type SubvectorMut = Self::Subvector;

    fn at_mut(&mut self, index: usize) -> &mut T {
        self.assert_in_range(index);
        self.data.at_mut(index + self.from)
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.assert_in_range(i);
        self.assert_in_range(j);
        self.data.swap(i + self.from, j + self.from);
    }

    fn cast_subvector(subvector: Self::Subvector) -> Self::SubvectorMut {
        subvector
    }
}

impl<T> VectorView<T> for Vec<T> {

    type Subvector = SubvectorView<Vec<T>>;

    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn at(&self, index: usize) -> &T {
        &self[index]
    }

    fn create_subvector(self, from: usize, to: usize) -> Self::Subvector {
        self.assert_in_range(from);
        assert!(to <= self.len());
        assert!(from <= to);
        SubvectorView::new(self, from, to)
    }
}

impl<T> VectorViewMut<T> for Vec<T> {

    type SubvectorMut = Self::Subvector;

    fn at_mut(&mut self, index: usize) -> &mut T {
        self.assert_in_range(index);
        &mut self[index]
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.assert_in_range(i);
        self.assert_in_range(j);
        <[T]>::swap(&mut self[..], i, j);
    }

    fn cast_subvector(subvector: Self::Subvector) -> Self::SubvectorMut {
        subvector
    }
}