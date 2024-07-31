use std::{cell::RefCell, cmp::Ordering, collections::BTreeMap, marker::PhantomData};

pub trait CompareFnFamily<K> {
    type CompareFn<'a>: for<'b> Fn(&'b K, &'b K) -> Ordering
        where Self: 'a;
}

struct CmpFunction<'a, K, F>
    where F: 'a + CompareFnFamily<K>
{
    f: F::CompareFn<'a>,
    family: PhantomData<F>,
    item: PhantomData<K>
}

impl<'a, K, F> CmpFunction<'a, K, F>
    where F: CompareFnFamily<K>
{
    fn from(f: F::CompareFn<'a>) -> Self {
        Self {
            f: f,
            family: PhantomData,
            item: PhantomData
        }
    }
}

thread_local!{
    static ENV: RefCell<Vec<*const ()>> = RefCell::new(Vec::new());
}

#[repr(transparent)]
struct Key<K, F>
    where F: CompareFnFamily<K>
{
    key: K,
    compare: PhantomData<F>
}

impl<K, F> Key<K, F>
    where F: CompareFnFamily<K>
{
    fn from(key: K) -> Self {
        Self {
            key: key,
            compare: PhantomData
        }
    }

    fn from_ref<'a>(key: &'a K) -> &'a Self {
        assert_eq!(std::mem::size_of::<Self>(), std::mem::size_of::<K>());
        assert_eq!(std::mem::align_of::<Self>(), std::mem::align_of::<K>());
        unsafe { std::mem::transmute(key)}
    }
}

impl<K, F> Ord for Key<K, F>
    where F: CompareFnFamily<K>
{
    fn cmp(&self, other: &Self) -> Ordering {
        ENV.with_borrow(|env| {
            let entry = env.last().unwrap();
            let cmp_fn = unsafe { &*std::mem::transmute::<*const (), *const CmpFunction<K, F>>(*entry) };
            (cmp_fn.f)(&self.key, &other.key)
        })
    }
}

impl<K, F> PartialOrd for Key<K, F>
    where F: CompareFnFamily<K>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K, F> PartialEq for Key<K, F>
    where F: CompareFnFamily<K>
{
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<K, F> Eq for Key<K, F>
    where F: CompareFnFamily<K>
{}

pub struct ExtCmpBTreeMap<K, V, F>
    where F: CompareFnFamily<K>
{
    map: BTreeMap<Key<K, F>, V>
}

impl<K, V, F> ExtCmpBTreeMap<K, V, F>
    where F: CompareFnFamily<K>
{
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new()
        }
    }
    
    fn perform_map_op<'a, 'b, G, T>(&'a self, f: G, cmp: F::CompareFn<'b>) -> T
        where G: FnOnce(&'a BTreeMap<Key<K, F>, V>) -> T
    {
        let cmp_fn = CmpFunction::from(cmp);
        ENV.with_borrow_mut(|env| {
            env.push(&cmp_fn as *const CmpFunction<K, F> as *const ());
        });
        let result = f(&self.map);
        ENV.with_borrow_mut(|env| {
            let popped = env.pop();
            assert!(std::ptr::eq(popped.unwrap(), &cmp_fn as *const CmpFunction<K, F> as *const ()));
        });
        return result;
    }

    fn perform_map_op_mut<'a, 'b, G, T>(&'a mut self, f: G, cmp: F::CompareFn<'b>) -> T
        where G: FnOnce(&'a mut BTreeMap<Key<K, F>, V>) -> T
    {
        let cmp_fn = CmpFunction::from(cmp);
        ENV.with_borrow_mut(|env| {
            env.push(&cmp_fn as *const CmpFunction<K, F> as *const ());
        });
        let result = f(&mut self.map);
        ENV.with_borrow_mut(|env| {
            let popped = env.pop();
            assert!(std::ptr::eq(popped.unwrap(), &cmp_fn as *const CmpFunction<K, F> as *const ()));
        });
        return result;
    }

    pub fn get<'a, 'b>(&'a self, key: &K, cmp: F::CompareFn<'b>) -> Option<&'a V> {
        self.perform_map_op(|map| map.get(Key::from_ref(key)), cmp)
    }

    pub fn insert<'b>(&mut self, key: K, value: V, cmp: F::CompareFn<'b>) -> Option<V> {
        self.perform_map_op_mut(|map| map.insert(Key::from(key), value), cmp)
    }
}

#[cfg(test)]
struct FnPtrCompareFnFamily<K> {
    key: PhantomData<K>
}

#[cfg(test)]
impl<K> CompareFnFamily<K> for FnPtrCompareFnFamily<K> {

    type CompareFn<'a> = &'a dyn Fn(&K, &K) -> Ordering
        where Self: 'a;
}

#[test]
fn test_extcmpmap() {
    fn bitrev_cmp(a: &u64, b: &u64) -> Ordering {
        a.reverse_bits().cmp(&b.reverse_bits())
    }
    let cmp_fn: &dyn Fn(&u64, &u64) -> Ordering = &bitrev_cmp;
    let mut map: ExtCmpBTreeMap<u64, bool, FnPtrCompareFnFamily<u64>> = ExtCmpBTreeMap::new();
    map.insert(6, false, cmp_fn);
    map.insert(3, true, cmp_fn);
    map.insert(4, false, cmp_fn);
    map.insert(1, true, cmp_fn);
    map.insert(2, false, cmp_fn);
    map.insert(5, true, cmp_fn);
    assert_eq!(Some(&true), map.get(&1, cmp_fn));
    assert_eq!(Some(&false), map.get(&2, cmp_fn));
    assert_eq!(Some(&true), map.get(&3, cmp_fn));
    assert_eq!(Some(&false), map.get(&4, cmp_fn));
    assert_eq!(Some(&true), map.get(&5, cmp_fn));
    assert_eq!(Some(&false), map.get(&6, cmp_fn));
    assert_eq!(None, map.get(&0, cmp_fn));
    assert_eq!(None, map.get(&7, cmp_fn));
}