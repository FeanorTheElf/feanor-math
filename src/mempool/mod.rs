use std::{ops::{Deref, DerefMut}, mem::MaybeUninit};

use crate::vector::VectorViewMut;

pub mod caching;

///
/// Trait for objects that can provide memory, often for short-term use.
/// This includes naive implementations like [`AllocatingMemoryProvider`] that
/// just allocate memory, or alternatively all kinds of memory pools and recyclers
/// (e.g. [`caching::CachingMemoryProvider`]).
/// 
/// This is related to [`std::alloc::Allocator`], but less restrictive, as it may
/// return objects with certain structure. In particular, it naturally allows e.g.
/// memory pools or memory recycling.
/// 
/// It is usually used when certain objects or algorithms need frequent allocations
/// (often all of the same size), either because they need temporary, internal memory,
/// or they represent rings and have to allocate memory for elements. On the other hand,
/// if a struct just needs to store some data during its lifetime, memory pooling is
/// usually not useful, and a standard `Vec` is often used instead.
/// 
pub trait MemoryProvider<T> {

    type Object: Deref<Target = [T]> + DerefMut + VectorViewMut<T>;

    unsafe fn get_new<F: FnOnce(&mut [MaybeUninit<T>])>(&self, size: usize, initializer: F) -> Self::Object;

    fn get_new_init<F: FnMut(usize) -> T>(&self, size: usize, mut initializer: F) -> Self::Object {
        unsafe {
            self.get_new(size, |mem| {
                for i in 0..mem.len() {
                    mem[i] = MaybeUninit::new(initializer(i))
                }
            })
        }
    }

    fn try_get_new_init<E, F: FnMut(usize) -> Result<T, E>>(&self, size: usize, mut initializer: F) -> Result<Self::Object, E> {
        unsafe {
            let mut aborted = None;
            let result = self.get_new(size, |mem| {
                let mut i = 0;
                while i < mem.len() {
                    // note that this will leak memory if initializer(i) panics
                    match initializer(i) {
                        Ok(val) => {
                            mem[i] = MaybeUninit::new(val);
                            i += 1;
                        },
                        Err(err) => {
                            aborted = Some(err);
                            // drop the previously initialized memory
                            // note that this does not prevent a memory leak in the panic case
                            for j in 0..i {
                                mem[j].assume_init_drop();
                            }
                            break;
                        }
                    };
                }
            });
            if let Some(err) = aborted {
                Err(err)
            } else {
                Ok(result)
            }
        }
    }
}

pub trait GrowableMemoryProvider<T>: MemoryProvider<T> {

    fn shrink(&self, el: &mut Self::Object, new_len: usize);

    unsafe fn grow<F: FnOnce(&mut [MaybeUninit<T>])>(&self, el: &mut Self::Object, new_size: usize, initializer: F);

    fn grow_init<F: FnMut(usize) -> T>(&self, el: &mut Self::Object, new_size: usize, mut initializer: F) {
        assert!(new_size > el.len());
        let old_len = el.len();
        unsafe {
            self.grow(el, new_size, |mem| {
                for i in 0..mem.len() {
                    mem[i] = MaybeUninit::new(initializer(old_len + i))
                }
            })
        }
    }

}

#[derive(Copy, Clone)]
pub struct AllocatingMemoryProvider;

impl<T> MemoryProvider<T> for AllocatingMemoryProvider {
    
    type Object = Vec<T>;

    unsafe fn get_new<F: FnOnce(&mut [MaybeUninit<T>])>(&self, size: usize, initializer: F) -> Self::Object {
        let mut result = Box::new_uninit_slice(size);
        initializer(&mut *result);
        return result.assume_init().into_vec();
    }
}

impl<T> GrowableMemoryProvider<T> for AllocatingMemoryProvider {
    
    unsafe fn grow<F: FnOnce(&mut [MaybeUninit<T>])>(&self, el: &mut Vec<T>, new_size: usize, initializer: F) {
        assert!(new_size > el.len());
        let old_len = el.len();
        el.reserve(new_size - old_len);
        initializer(&mut el.spare_capacity_mut()[..(new_size - old_len)]);
        el.set_len(new_size);
    }

    fn shrink(&self, el: &mut Self::Object, new_len: usize) {
        el.truncate(new_len)
    }
}

pub static ALLOCATING_MEMORY_PROVIDER_SINGLETON: AllocatingMemoryProvider = AllocatingMemoryProvider;

impl Default for AllocatingMemoryProvider {

    fn default() -> Self {
        AllocatingMemoryProvider
    }
}

#[derive(Clone, Copy)]
pub struct LoggingMemoryProvider {
    description: &'static str
}

impl LoggingMemoryProvider {

    pub const fn new(description: &'static str) -> Self {
        LoggingMemoryProvider { description }
    }
}

impl<'a, T> MemoryProvider<T> for &'a LoggingMemoryProvider {
    
    type Object = Vec<T>;

    unsafe fn get_new<F: FnOnce(&mut [MaybeUninit<T>])>(&self, size: usize, initializer: F) -> Self::Object {
        println!("[{}]: Allocating {} entries of size {}", self.description, size, std::mem::size_of::<T>());
        AllocatingMemoryProvider.get_new(size, initializer)
    }
}

impl<'a, T> GrowableMemoryProvider<T> for &'a LoggingMemoryProvider {
    
    unsafe fn grow<F: FnOnce(&mut [MaybeUninit<T>])>(&self, el: &mut Vec<T>, new_size: usize, initializer: F) {
        assert!(new_size > el.len());
        let old_len = el.len();
        el.reserve(new_size - old_len);
        initializer(&mut el.spare_capacity_mut()[..(new_size - old_len)]);
        el.set_len(new_size);
    }

    fn shrink(&self, el: &mut Self::Object, new_len: usize) {
        el.truncate(new_len)
    }
}

#[cfg(not(feature = "log_memory"))]
pub type DefaultMemoryProvider = AllocatingMemoryProvider;

#[cfg(feature = "log_memory")]
pub type DefaultMemoryProvider = &'static LoggingMemoryProvider;

#[macro_export]
macro_rules! current_function {
    () => {{
        struct LocalMemoryProvider;
        std::any::type_name::<LocalMemoryProvider>()
    }}
}

#[macro_export]
#[cfg(not(feature = "log_memory"))]
macro_rules! default_memory_provider {
    () => {
        $crate::mempool::ALLOCATING_MEMORY_PROVIDER_SINGLETON
    };
}

#[macro_export]
#[cfg(feature = "log_memory")]
macro_rules! default_memory_provider {
    () => {
        {
            static LOCAL_MEMORY_PROVIDER: $crate::mempool::LoggingMemoryProvider = $crate::mempool::LoggingMemoryProvider::new($crate::current_function!());
            &LOCAL_MEMORY_PROVIDER as &'static $crate::mempool::LoggingMemoryProvider
        }
    };
}