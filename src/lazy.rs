use std::sync::OnceLock;

use append_only_vec::AppendOnlyVec;

///
/// Vector whose elements are never modified but added.
/// More general than [`AppendOnlyVec`] in that we can control
/// at what index an element is inserted, if not present.
/// 
pub struct LazyVec<T> {
    data: AppendOnlyVec<OnceLock<T>>
}

impl<T> LazyVec<T> {

    pub fn new() -> Self {
        Self {
            data: AppendOnlyVec::new()
        }
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        self.data[i].get()
    }

    pub fn get_or_init_incremental<F>(&self, target_len: usize, mut derive_next: F)
        where F: FnMut(usize, &T) -> T
    {
        while self.data.len() <= target_len {
            _ = self.data.push(OnceLock::new());
        }
        for i in (0..=target_len).rev() {
            if let Some(mut value) = self.data[i].get() {
                for j in (i + 1)..=target_len {
                    value = self.data[j].get_or_init(|| derive_next(j, value));
                }
                return;
            }
        }
        panic!("get_or_init_incremental() is only valid when the vector has at least one initialized element")
    }

    pub fn get_or_init<'a, F>(&'a self, i: usize, init: F) -> &'a T
        where F: FnOnce() -> T
    {
        while self.data.len() <= i {
            _ = self.data.push(OnceLock::new());
        }
        return self.data[i].get_or_init(init);
    }
}

impl<T> Clone for LazyVec<T>
    where T: Clone
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.iter().map(|data| if let Some(value) = data.get() {
                let result = OnceLock::new();
                _ = result.set(value.clone());
                result
            } else {
                OnceLock::new()
            }).collect()
        }
    }
}