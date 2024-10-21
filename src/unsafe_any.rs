use std::{alloc::{Allocator, Global, Layout}, ptr::NonNull};

///
/// Like [`Box`] but not generic in the type it stores.
/// This type has to be provided when creating or accessing the container.
/// If there is a mismatch between the provided types, it causes UB.
/// 
/// I mainly use this in specific circumstances as a workaround for
/// specializing associated types.
/// 
pub struct UnsafeAny {
    data: Option<NonNull<[u8]>>,
    deleter: unsafe fn(NonNull<[u8]>)
}

unsafe fn delete<T>(value: NonNull<[u8]>) {
    // that is basicaly the only check we are able to do...
    assert_eq!(std::mem::size_of::<T>(), value.as_ref().len());
    let ptr = std::mem::transmute::<*const (), *const T>(value.as_ptr() as *const ());
    std::ptr::read(ptr);
}

impl UnsafeAny {

    pub fn uninit() -> UnsafeAny {
        UnsafeAny { data: None, deleter: delete::<()> }
    }
    
    pub unsafe fn from<T>(value: T) -> UnsafeAny {
        unsafe { 
            let memory = Global.allocate_zeroed(Layout::for_value(&value)).unwrap();
            assert_eq!(std::mem::size_of::<T>(), memory.len());
            std::ptr::write(std::mem::transmute::<*mut (), *mut T>(memory.as_ptr() as *mut ()), value);
            UnsafeAny { data: Some(memory), deleter: delete::<T> }
        }
    }

    pub unsafe fn get<'a, T>(&'a self) -> &'a T {
        assert!(self.data.is_some());
        // that is basicaly the only check we are able to do...
        assert_eq!(std::mem::size_of::<T>(), self.data.unwrap().as_ref().len());
        &*std::mem::transmute::<*const (), *const T>(self.data.unwrap().as_ptr() as *const ())
    }
}

impl Drop for UnsafeAny {

    fn drop(&mut self) {
        if let Some(ptr) = self.data {
            unsafe { (self.deleter)(ptr) }
        }
    }
}
