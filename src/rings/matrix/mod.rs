use std::cmp::Ordering;
use std::marker::PhantomData;

use crate::mempool::{MemoryProvider, GrowableMemoryProvider};
use crate::ring::*;

const DENSITY_RATIO_FACTOR: f64 = 2.;
const DENSITY_RATIO_UPDATE: f64 = 0.001;

struct ColMajorSparseMatrix<T, MCol, MRow>
    where MCol: GrowableMemoryProvider<(T, usize)>,
        MRow: MemoryProvider<MCol::Object>
{
    el: PhantomData<T>,
    cols: MRow::Object
}

struct RowMajorSparseMatrix<T, MRow, MCol>
    where MRow: GrowableMemoryProvider<(T, usize)>,
        MCol: MemoryProvider<MRow::Object>
{
    el: PhantomData<T>,
    rows: MCol::Object
}

fn inner_product<R>(ring: R, lhs: &[(El<R>, usize)], rhs: &[(El<R>, usize)]) -> El<R>
    where R: RingStore
{
    let mut li = 0;
    let mut ri = 0;
    let mut current = ring.zero();
    while li < lhs.len() && ri < rhs.len() {
        match lhs[li].1.cmp(&rhs[ri].1) {
            Ordering::Less => { li += 1 },
            Ordering::Greater => { ri += 1 },
            Ordering::Equal => {
                ring.add_assign(&mut current, ring.mul_ref(&lhs[li].0, &rhs[ri].0));
                li += 1;
                ri += 1;
            }
        }
    }
    return current;
}

fn mul_assign_left<R, MCol1, MRow1, MRow2, MCol2>(ring: R, out: &mut ColMajorSparseMatrix<El<R>, MCol1, MRow1>, factor: &RowMajorSparseMatrix<El<R>, MRow2, MCol2>, density_ratio_estimate: &mut f64) 
    where R: RingStore,
        MCol1: GrowableMemoryProvider<(El<R>, usize)>,
        MRow1: MemoryProvider<MCol1::Object>,
        MRow2: GrowableMemoryProvider<(El<R>, usize)>,
        MCol2: MemoryProvider<MRow2::Object>
{
    let n = out.cols.len();
    for j in 0..n {

    }
}