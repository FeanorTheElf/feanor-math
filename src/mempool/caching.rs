use std::{mem::MaybeUninit, rc::Rc, collections::{HashMap, hash_map::Entry, HashSet}, ops::{Deref, DerefMut}, cell::{RefCell, Ref}, sync::{Mutex, atomic::AtomicBool}, hash::Hash};

use super::{MemoryProvider, AllocatingMemoryProvider};

pub struct CachedMemoryData<T: Sized> {
    data: Option<Box<[MaybeUninit<T>]>>,
    return_to: Rc<CachingMemoryProvider<T>>,
}

impl<T: Sized> Deref for CachedMemoryData<T> {

    type Target = [T];

    fn deref<'a>(&'a self) -> &'a Self::Target {
        unsafe {
            MaybeUninit::slice_assume_init_ref(self.data.as_ref().unwrap())
        }
    }
}

impl<T: Sized> DerefMut for CachedMemoryData<T> {

    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            MaybeUninit::slice_assume_init_mut(self.data.as_mut().unwrap())
        }
    }
}

impl<T: Sized> Drop for CachedMemoryData<T> {

    fn drop(&mut self) {
        unsafe {
            for i in 0..self.len() {
                self.data.as_mut().unwrap()[i].assume_init_drop();
            }
            self.return_to.put_back(self.data.take().unwrap())
        }
    }
}

pub struct CachingMemoryProvider<T: Sized> {
    stored_arrays: RefCell<HashMap<usize, Vec<Box<[MaybeUninit<T>]>>>>,
    max_stored: usize
}

impl<T: Sized> CachingMemoryProvider<T> {

    pub fn new(max_store_per_size: usize) -> Rc<Self> {
        Rc::new(Self {
            stored_arrays: RefCell::new(HashMap::new()),
            max_stored: max_store_per_size
        })
    }

    fn put_back(&self, element: Box<[MaybeUninit<T>]>) {
        let mut locked = self.stored_arrays.borrow_mut();
        match locked.entry(element.len()) {
            Entry::Occupied(mems) if mems.get().len() >= self.max_stored => {},
            Entry::Occupied(mut mems) => mems.get_mut().push(element),
            Entry::Vacant(mems) => { mems.insert(vec![element]); }
        };
    }
}

impl<T: Sized> MemoryProvider<T> for Rc<CachingMemoryProvider<T>> {

    type Object = CachedMemoryData<T>;

    unsafe fn get_new<F: FnOnce(&mut [MaybeUninit<T>])>(&self, size: usize, initializer: F) -> Self::Object {
        let mut locked = self.stored_arrays.borrow_mut();
        if let Some(mems) = locked.get_mut(&size) {
            if let Some(mut result) = mems.pop() {
                initializer(&mut result);
                return CachedMemoryData {
                    data: Some(result),
                    return_to: self.clone()
                };
            }
        }
        let mut result = Box::new_uninit_slice(size);
        initializer(&mut *result);
        return CachedMemoryData {
            data: Some(result),
            return_to: self.clone()
        };
    }
}

#[test]
fn test_caching_memory_provider() {
    let mem_provider = CachingMemoryProvider::new(2);
    let b_ptr = {
        let a = mem_provider.get_new_init(3, |_| 0);
        let b = mem_provider.get_new_init(3, |_| 1);
        let c = mem_provider.get_new_init(3, |_| 2);
        assert_eq!(0, a[0]);
        assert_eq!(1, b[1]);
        assert_eq!(2, c[1]);
        (&*b).as_ptr()
    };
    let c = mem_provider.get_new_init(3, |_| 2);
    assert_eq!((&*c).as_ptr(), b_ptr);
    assert_eq!(2, c[1]);
}

#[test]
fn test_caching_memory_provider_drop_exactly_once() {

    struct Test(i32);

    static DROPPED: Mutex<Option<HashSet<i32>>> = Mutex::new(None);
    static FAILED: AtomicBool = AtomicBool::new(false);

    *DROPPED.lock().unwrap() = Some(HashSet::new());

    impl Drop for Test {

        fn drop(&mut self) {
            let mut locked = DROPPED.lock().unwrap();
            let mut locked = locked.as_mut().unwrap();
            if locked.contains(&self.0) {
                FAILED.fetch_or(true, std::sync::atomic::Ordering::SeqCst);
            } else {
                locked.insert(self.0);
            }
        }
    }
    {
        let mem_provider = CachingMemoryProvider::new(2);
        {
            let a = mem_provider.get_new_init(1, |_| Test(0));
            let b = mem_provider.get_new_init(1, |_| Test(1));
            let c = mem_provider.get_new_init(1, |_| Test(2));

            let a = mem_provider.get_new_init(2, |i| Test(6 + i as i32));
            let b = mem_provider.get_new_init(2, |i| Test(8 + i as i32));
            let c = mem_provider.get_new_init(2, |i| Test(10 + i as i32));
        }
        {
            let a = mem_provider.get_new_init(1, |_| Test(3));
            let b = mem_provider.get_new_init(1, |_| Test(4));
            let c = mem_provider.get_new_init(1, |_| Test(5));
            
            let a = mem_provider.get_new_init(2, |i| Test(12 + i as i32));
            let b = mem_provider.get_new_init(2, |i| Test(14 + i as i32));
            let c = mem_provider.get_new_init(2, |i| Test(16 + i as i32));
        }
    }

    assert!(!FAILED.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(*DROPPED.lock().unwrap().as_ref().unwrap(), (0..18).collect());
}