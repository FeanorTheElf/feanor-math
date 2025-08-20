use serde::de::*;
use serde::{Deserializer, Serialize, Serializer};

use crate::ring::*;

///
/// Trait for rings whose elements can be serialized.
/// 
/// Serialization and deserialization mostly follow the principles of the `serde` crate, with
/// the main difference that ring elements cannot be serialized/deserialized on their own, but
/// only w.r.t. a specific ring.
/// 
#[stability::unstable(feature = "enable")]
pub trait SerializableElementRing: RingBase {

    ///
    /// Deserializes an element of this ring from the given deserializer.
    /// 
    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de>;

    ///
    /// Serializes an element of this ring to the given serializer.
    /// 
    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer;
}


///
/// Wrapper of a ring that implements [`serde::DeserializationSeed`] by trying to deserialize an element
/// w.r.t. the wrapped ring.
/// 
#[stability::unstable(feature = "enable")]
#[derive(Clone)]
pub struct DeserializeWithRing<R: RingStore>
    where R::Type: SerializableElementRing
{
    ring: R
}

impl<R> DeserializeWithRing<R>
    where R::Type: SerializableElementRing,
        R: RingStore
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: R) -> Self {
        Self { ring }
    }
}

impl<'de, R> DeserializeSeed<'de> for DeserializeWithRing<R>
    where R::Type: SerializableElementRing,
        R: RingStore
{
    type Value = El<R>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where D: Deserializer<'de>
    {
        self.ring.get_ring().deserialize(deserializer)
    }
}

///
/// Wraps a ring and a reference to one of its elements. Implements [`serde::Serialize`] and
/// will serialize the element w.r.t. the ring.
/// 
#[stability::unstable(feature = "enable")]
pub struct SerializeWithRing<'a, R: RingStore>
    where R::Type: SerializableElementRing
{
    ring: R,
    el: &'a El<R>
}

impl<'a, R: RingStore> SerializeWithRing<'a, R>
    where R::Type: SerializableElementRing
{
    #[stability::unstable(feature = "enable")]
    pub fn new(el: &'a El<R>, ring: R) -> Self {
        Self { el, ring }
    }
}

impl<'a, R: RingStore> Serialize for SerializeWithRing<'a, R>
    where R::Type: SerializableElementRing
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        self.ring.get_ring().serialize(self.el, serializer)
    }
}

///
/// Wraps a ring and a one of its elements. Implements [`serde::Serialize`] and
/// will serialize the element w.r.t. the ring.
/// 
#[stability::unstable(feature = "enable")]
pub struct SerializeOwnedWithRing<R: RingStore>
    where R::Type: SerializableElementRing
{
    ring: R,
    el: El<R>
}

impl<R: RingStore> SerializeOwnedWithRing<R>
    where R::Type: SerializableElementRing
{
    #[stability::unstable(feature = "enable")]
    pub fn new(el: El<R>, ring: R) -> Self {
        Self { el, ring }
    }
}

impl<R: RingStore> Serialize for SerializeOwnedWithRing<R>
    where R::Type: SerializableElementRing
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        self.ring.get_ring().serialize(&self.el, serializer)
    }
}

#[stability::unstable(feature = "enable")]
#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {

    use super::*;

    #[stability::unstable(feature = "enable")]
    pub fn test_serialization<R: RingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I)
        where R::Type: SerializableElementRing
    {
        let edge_case_elements = edge_case_elements.collect::<Vec<_>>();

        let serializer = serde_assert::Serializer::builder().is_human_readable(true).build();
        for x in &edge_case_elements {
            let tokens = ring.get_ring().serialize(&x, &serializer).unwrap();
            let mut deserializer = serde_assert::Deserializer::builder(tokens).is_human_readable(true).build();
            let result = ring.get_ring().deserialize(&mut deserializer).unwrap();
            assert_el_eq!(ring, &result, &x);
        }

        let serializer = serde_assert::Serializer::builder().is_human_readable(false).build();
        for x in &edge_case_elements {
            let tokens = ring.get_ring().serialize(&x, &serializer).unwrap();
            let mut deserializer = serde_assert::Deserializer::builder(tokens).is_human_readable(false).build();
            let result = ring.get_ring().deserialize(&mut deserializer).unwrap();
            assert_el_eq!(ring, &result, &x);
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn test_serialize_deserialize<T: Serialize + for<'de> Deserialize<'de> + PartialEq>(x: T) {
        let serializer = serde_assert::Serializer::builder().is_human_readable(true).build();
        let tokens = x.serialize(&serializer).unwrap();
        let mut deserializer = serde_assert::Deserializer::builder(tokens).is_human_readable(true).build();
        let result = T::deserialize(&mut deserializer).unwrap();
        assert!(result == x);

        let serializer = serde_assert::Serializer::builder().is_human_readable(false).build();
        let tokens = x.serialize(&serializer).unwrap();
        let mut deserializer = serde_assert::Deserializer::builder(tokens).is_human_readable(false).build();
        let result = T::deserialize(&mut deserializer).unwrap();
        assert!(result == x);
    }
}

#[cfg(test)]
use crate::integer::{BigIntRing, IntegerRingStore};

#[test]
fn test_serialize() {
    let value = BigIntRing::RING.add(BigIntRing::RING.power_of_two(128), BigIntRing::RING.one());
    let json = serde_json::to_string(&SerializeWithRing::new(&value, BigIntRing::RING)).unwrap();
    assert_eq!("\"340282366920938463463374607431768211457\"", json);
}