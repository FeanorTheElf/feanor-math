use std::{mem::MaybeUninit, rc::Rc, collections::HashMap, ops::{Deref, DerefMut}};

use super::MemoryProvider;

pub struct CachedMemoryData<T: Sized> {
    data: Box<[MaybeUninit<T>]>,
    return_to: Rc<CachingMemoryProvider<T>>,
}

impl<T: Sized> Deref for CachedMemoryData<T> {

    type Target = [T];

    fn deref<'a>(&'a self) -> &'a Self::Target {
        unsafe {
            MaybeUninit::slice_assume_init_ref(&*self.data)
        }
    }
}

impl<T: Sized> DerefMut for CachedMemoryData<T> {

    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            MaybeUninit::slice_assume_init_mut(&mut *self.data)
        }
    }
}

impl<T: Sized> Drop for CachedMemoryData<T> {

    fn drop(&mut self) {
        unsafe {
            for i in 0..self.len() {
                self.data[i].assume_init_drop();
            }
        }
    }
}

pub struct CachingMemoryProvider<T: Sized> {
    stored_arrays: HashMap<usize, Vec<Box<[MaybeUninit<T>]>>>,
    max_stored: usize
}

impl<T: Sized> MemoryProvider<T> for CachingMemoryProvider<T> {
}

impl<T: Sized> MemoryProvider<T> for CachingMemoryProvider<T> {

    type Object = CachedMemoryData<T>;

    unsafe fn get_new<F: FnOnce(&mut [MaybeUninit<T>])>(&self, size: usize, initializer: F) -> Self::Object {
        
    }
}