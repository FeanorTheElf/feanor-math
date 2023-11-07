use std::marker::PhantomData;

use crate::mempool::{MemoryProvider, GrowableMemoryProvider};
use crate::ring::*;

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

fn mul_assign_left<R, MCol1, MRow1, MRow2, MCol2>(ring: R, out: &mut ColMajorSparseMatrix<El<R>, MCol1, MRow1>, factor: &RowMajorSparseMatrix<El<R>, MRow2, MCol2>) 
    where R: RingStore,
        MCol1: GrowableMemoryProvider<(El<R>, usize)>,
        MRow1: MemoryProvider<MCol1::Object>,
        MRow2: GrowableMemoryProvider<(El<R>, usize)>,
        MCol2: MemoryProvider<MRow2::Object>
{

}