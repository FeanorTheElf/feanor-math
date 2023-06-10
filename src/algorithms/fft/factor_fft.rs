use crate::{ring::*, mempool::*};
use crate::algorithms::fft::*;

pub struct FFTTableGenCooleyTuckey<R, T1, T2, M = AllocatingMemoryProvider> 
    where R: RingStore,
        T1: FFTTable<Ring = R>,
        T2: FFTTable<Ring = R>,
        M: MemoryProvider<El<R>>
{
    twiddle_factors: M::Object,
    inv_twiddle_factors: M::Object,
    left_table: T1,
    right_table: T2,
    root_of_unity: El<R>
}

impl<R, T1, T2> FFTTableGenCooleyTuckey<R, T1, T2>
    where R: RingStore,
        T1: FFTTable<Ring = R>,
        T2: FFTTable<Ring = R>
{
    pub fn new(root_of_unity: El<R>, left_table: T1, right_table: T2) -> Self {
        Self::new_with_mem(root_of_unity, left_table, right_table, &AllocatingMemoryProvider)
    }
}
impl<R, T1, T2, M> FFTTableGenCooleyTuckey<R, T1, T2, M>
    where R: RingStore,
        T1: FFTTable<Ring = R>,
        T2: FFTTable<Ring = R>,
        M: MemoryProvider<El<R>>
{
    pub fn new_with_mem(root_of_unity: El<R>, left_table: T1, right_table: T2, memory_provider: &M) -> Self {
        assert!(left_table.ring().get_ring() == right_table.ring().get_ring());

        let ring = left_table.ring();
        assert!(ring.get_ring().is_approximate() || ring.eq_el(&ring.pow(ring.clone_el(&root_of_unity), right_table.len()), left_table.root_of_unity()));
        assert!(ring.get_ring().is_approximate() || ring.eq_el(&ring.pow(ring.clone_el(&root_of_unity), left_table.len()), right_table.root_of_unity()));

        let inv_root_of_unity = ring.pow(ring.clone_el(&root_of_unity), right_table.len() * left_table.len() - 1);
        let inv_twiddle_factors = memory_provider.get_new_init(left_table.len() * right_table.len(), |i| {
            let ri = i % right_table.len();
            let li = i / right_table.len();
            return ring.pow(ring.clone_el(&inv_root_of_unity), left_table.unordered_fft_permutation(li) * ri);
        });
        let twiddle_factors = memory_provider.get_new_init(left_table.len() * right_table.len(), |i| {
            let ri = i % right_table.len();
            let li = i / right_table.len();
            return ring.pow(ring.clone_el(&root_of_unity), left_table.unordered_fft_permutation(li) * ri);
        });

        FFTTableGenCooleyTuckey {
            twiddle_factors: twiddle_factors,
            inv_twiddle_factors: inv_twiddle_factors,
            left_table: left_table, 
            right_table: right_table,
            root_of_unity: root_of_unity
        }
    }
}

impl<R, T1, T2, M> FFTTable for FFTTableGenCooleyTuckey<R, T1, T2, M>
    where R: RingStore,
        T1: FFTTable<Ring = R>,
        T2: FFTTable<Ring = R>,
        M: MemoryProvider<El<R>>
{
    type Ring = R;

    fn len(&self) -> usize {
        self.left_table.len() * self.right_table.len()
    }

    fn ring(&self) -> &R {
        self.left_table.ring()
    }

    fn root_of_unity(&self) -> &El<R> {
        &self.root_of_unity
    }

    fn unordered_fft<V, S, N>(&self, mut values: V, ring: S, memory_provider: &N)
        where S: RingStore,
            S::Type: CanonicalHom<<R as RingStore>::Type>,
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        let hom = ring.can_hom(self.ring()).unwrap();
        for i in 0..self.right_table.len() {
            let mut v = Subvector::new(&mut values).subvector(i..).stride(self.right_table.len());
            self.left_table.unordered_fft(&mut v, &ring, memory_provider);
        }
        for i in 0..self.len() {
            ring.get_ring().mul_assign_map_in_ref(self.ring().get_ring(), values.at_mut(i), self.inv_twiddle_factors.at(i), hom.raw_hom());
        }
        for i in 0..self.left_table.len() {
            let mut v = Subvector::new(&mut values).subvector((i * self.right_table.len())..((i + 1) * self.right_table.len()));
            self.right_table.unordered_fft(&mut v, &ring, memory_provider);
        }
    }

    fn unordered_inv_fft<V, S, N>(&self, mut values: V, ring: S, memory_provider: &N)
        where S: RingStore,
            S::Type: CanonicalHom<<R as RingStore>::Type>,
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        let hom = ring.can_hom(self.ring()).unwrap();
        for i in 0..self.left_table.len() {
            let mut v = Subvector::new(&mut values).subvector((i * self.right_table.len())..((i + 1) * self.right_table.len()));
            self.right_table.unordered_inv_fft(&mut v, &ring, memory_provider);
        }
        for i in 0..self.len() {
            ring.get_ring().mul_assign_map_in_ref(self.ring().get_ring(), values.at_mut(i), self.twiddle_factors.at(i), hom.raw_hom());
        }
        for i in 0..self.right_table.len() {
            let mut v = Subvector::new(&mut values).subvector(i..).stride(self.right_table.len());
            self.left_table.unordered_inv_fft(&mut v, &ring, memory_provider);
        }
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        let ri = i % self.right_table.len();
        let li = i / self.right_table.len();
        return self.left_table.unordered_fft_permutation(li) + self.left_table.len() * self.right_table.unordered_fft_permutation(ri);
    }
}

#[cfg(test)]
use crate::rings::zn::zn_static::Zn;

#[test]
fn test_fft_basic() {
    let ring = Zn::<97>::RING;
    let z = ring.from_int(39);
    let fft = FFTTableGenCooleyTuckey::new(ring.pow(z, 16), 
        bluestein::FFTTableBluestein::new(ring, ring.pow(z, 24), ring.pow(z, 12), 2, 3),
        bluestein::FFTTableBluestein::new(ring, ring.pow(z, 16), ring.pow(z, 12), 3, 3),
    );
    let mut values = [1, 0, 0, 1, 0, 1];
    let expected = [3, 62, 63, 96, 37, 36];
    let mut permuted_expected = [0; 6];
    for i in 0..6 {
        permuted_expected[i] = expected[fft.unordered_fft_permutation(i)];
    }

    fft.unordered_fft(&mut values, ring, &AllocatingMemoryProvider);
    assert_eq!(values, permuted_expected);
}

#[test]
fn test_fft_long() {
    let ring = Zn::<97>::RING;
    let z = ring.from_int(39);
    let fft = FFTTableGenCooleyTuckey::new(ring.pow(z, 4), 
        bluestein::FFTTableBluestein::new(ring, ring.pow(z, 6), ring.pow(z, 3), 8, 5),
        bluestein::FFTTableBluestein::new(ring, ring.pow(z, 16), ring.pow(z, 12), 3, 3),
    );
    let mut values = [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 2, 2, 0, 2, 0, 1, 2, 3, 4];
    let expected = [26, 0, 75, 47, 41, 31, 28, 62, 39, 93, 53, 27, 0, 54, 74, 61, 65, 81, 63, 38, 53, 94, 89, 91];
    let mut permuted_expected = [0; 24];
    for i in 0..24 {
        permuted_expected[i] = expected[fft.unordered_fft_permutation(i)];
    }

    fft.unordered_fft(&mut values, ring, &AllocatingMemoryProvider);
    assert_eq!(values, permuted_expected);
}

#[test]
fn test_inv_fft() {
    let ring = Zn::<97>::RING;
    let z = ring.from_int(39);
    let fft = FFTTableGenCooleyTuckey::new(ring.pow(z, 16), 
        bluestein::FFTTableBluestein::new(ring, ring.pow(z, 24), ring.pow(z, 12), 2, 3),
        bluestein::FFTTableBluestein::new(ring, ring.pow(z, 16), ring.pow(z, 12), 3, 3),
    );
    let mut values = [3, 62, 63, 96, 37, 36];
    let expected = [1, 0, 0, 1, 0, 1];

    fft.inv_fft(&mut values, ring, &AllocatingMemoryProvider);
    assert_eq!(values, expected);
}
