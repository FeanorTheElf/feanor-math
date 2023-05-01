use std::{marker::PhantomData, ops::{Deref, DerefMut}, mem::MaybeUninit, rc::Rc};

pub struct MemoryObject<M: ?Sized, T, D = usize>
    where M: MemoryProviderBase<T, Data = D>
{
    data: MaybeUninit<D>,
    target_type: PhantomData<T>,
    memory_provider: Rc<M>
}

impl<M: ?Sized, T, D> Deref for MemoryObject<M, T, D>
    where M: MemoryProviderBase<T, Data = D>
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { 
            self.memory_provider.access(self.data.assume_init_ref())
        }
    }
}

impl<M: ?Sized, T, D> DerefMut for MemoryObject<M, T, D>
    where M: MemoryProviderBase<T, Data = D>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { 
            self.memory_provider.access_mut(self.data.assume_init_mut())
        }
    }
}

impl<M: ?Sized, T, D> Drop for MemoryObject<M, T, D>
    where M: MemoryProviderBase<T, Data = D>
{
    fn drop(&mut self) {
        unsafe {
            let self_data = std::mem::replace(&mut self.data, MaybeUninit::zeroed());
            self.memory_provider.put_back(self_data.assume_init());
        }
    }
}

pub trait MemoryProviderBase<T> {

    type Data;

    fn access<'a>(&'a self, data: &'a Self::Data) -> &'a mut [T];
    fn access_mut<'a>(&'a self, data: &'a mut Self::Data) -> &'a mut [T];
    fn put_back(&self, data: Self::Data);
    unsafe fn get_new<F: FnOnce(&mut [MaybeUninit<T>])>(&self, size: usize, initializer: F) -> Self::Data;
}

pub trait MemoryProvider<T> {

    type Object: Deref<Target = [T]> + DerefMut;

    unsafe fn get_new<F: FnOnce(&mut [MaybeUninit<T>])>(&self, size: usize, initializer: F) -> Self::Object;

    fn get_new_init<F: Fn() -> T>(&self, size: usize, initializer: F) -> Self::Object {
        unsafe {
            self.get_new(size, |mem| {
                for i in 0..mem.len() {
                    mem[i] = MaybeUninit::new(initializer())
                }
            })
        }
    }
}

impl<T, M: ?Sized> MemoryProvider<T> for Rc<M>
    where M: MemoryProviderBase<T>
{
    type Object = MemoryObject<M, T, M::Data>;

    unsafe fn get_new<F: FnOnce(&mut [MaybeUninit<T>])>(&self, size: usize, initializer: F) -> Self::Object {
        let data = self.deref().get_new(size, initializer);
        let self_copy = self.clone();
        MemoryObject {
            data: MaybeUninit::new(data),
            target_type: PhantomData,
            memory_provider: self_copy
        }
    }
}

pub struct AllocatingMemoryProvider;

impl<T> MemoryProvider<T> for AllocatingMemoryProvider {
    
    type Object = Box<[T]>;

    unsafe fn get_new<F: FnOnce(&mut [MaybeUninit<T>])>(&self, size: usize, initializer: F) -> Self::Object {
        let mut result = Box::new_uninit_slice(size);
        initializer(&mut *result);
        return result.assume_init();
    }
}

pub struct LoggingMemoryProvider {
    description: String
}

impl LoggingMemoryProvider {

    pub fn new(description: String) -> Self {
        LoggingMemoryProvider { description }
    }
}

impl<T> MemoryProvider<T> for LoggingMemoryProvider {
    
    type Object = Box<[T]>;

    unsafe fn get_new<F: FnOnce(&mut [MaybeUninit<T>])>(&self, size: usize, initializer: F) -> Self::Object {
        println!("[{}]: Allocating {} entries", self.description, size);
        AllocatingMemoryProvider.get_new(size, initializer)
    }
}