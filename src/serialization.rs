use std::marker::PhantomData;

use serde::de::{DeserializeSeed, IgnoredAny, SeqAccess, Visitor};
use serde::{Deserializer, Serialize, Serializer};

use crate::seq::VectorFn;

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
/// Wraps a [`VectorFn`] whose elements are [`Serialize`]able, and
/// allows to serialize all elements as a sequence.
/// 
/// To deserialize, consider using [`DeserializeSeedSeq`].
/// 
#[stability::unstable(feature = "enable")]
pub struct SerializableSeq<V, T>
    where V: VectorFn<T>
{
    element: PhantomData<T>,
    data: V
}

impl<V, T> SerializableSeq<V, T>
    where V: VectorFn<T>
{
    #[stability::unstable(feature = "enable")]
    pub fn new(data: V) -> Self {
        Self { element: PhantomData, data: data }
    }
}

impl<V, T> Serialize for SerializableSeq<V, T>
    where V: VectorFn<T>, T: Serialize
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        serializer.collect_seq(self.data.iter())
    }
}

///
/// A [`DeserializeSeed`] that deserializes a sequence by deserializing
/// every element according to some derived [`DeserializeSeed`]. The resulting
/// elements are collected using a custom reducer function, akin to 
/// [`Iterator::fold()`].
/// 
#[stability::unstable(feature = "enable")]
pub struct DeserializeSeedSeq<'de, V, S, T, C>
    where V: Iterator<Item = S>,
        S: DeserializeSeed<'de>,
        C: FnMut(T, S::Value) -> T
{
    deserializer: PhantomData<&'de ()>,
    element_seed: PhantomData<S>,
    seeds: V,
    initial: T,
    collector: C
}

impl<'de, V, S, T, C> DeserializeSeedSeq<'de, V, S, T, C>
    where V: Iterator<Item = S>,
        S: DeserializeSeed<'de>,
        C: FnMut(T, S::Value) -> T
{
    #[stability::unstable(feature = "enable")]
    pub fn new(seeds: V, initial: T, collector: C) -> Self {
        Self {
            deserializer: PhantomData,
            element_seed: PhantomData,
            seeds: seeds,
            initial: initial,
            collector: collector
        }
    }
}

impl<'de, V, S, T, C> DeserializeSeed<'de> for DeserializeSeedSeq<'de, V, S, T, C>
    where V: Iterator<Item = S>, 
        S: DeserializeSeed<'de>,
        C: FnMut(T, S::Value) -> T
{
    type Value = T;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where D: serde::Deserializer<'de>
    {
        struct ResultVisitor<'de, V, S, T, C>
            where V: Iterator<Item = S>,
                S: DeserializeSeed<'de>,
                C: FnMut(T, S::Value) -> T
        {
            deserializer: PhantomData<&'de ()>,
            element_seed: PhantomData<S>,
            seeds: V,
            initial: T,
            collector: C
        }

        impl<'de, V, S, T, C> Visitor<'de> for ResultVisitor<'de, V, S, T, C>
            where V: Iterator<Item = S>,
                S: DeserializeSeed<'de>,
                C: FnMut(T, S::Value) -> T
        {
            type Value = T;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "a sequence of elements")
            }

            fn visit_seq<B>(mut self, mut seq: B) -> Result<Self::Value, B::Error>
                where B: SeqAccess<'de>
            {
                let mut result = self.initial;
                for current in 0..usize::MAX {
                    if let Some(seed) = self.seeds.next() {
                        let el = seq.next_element_seed(seed)?;
                        if let Some(el) = el {
                            result = (self.collector)(result, el);
                        } else {
                            return Ok(result);
                        }
                    } else {
                        if let Some(_) = seq.next_element::<IgnoredAny>()? {
                            return Err(<B::Error as serde::de::Error>::invalid_length(current + 1, &format!("a sequence of at most {} elements", current).as_str()));
                        } else {
                            return Ok(result);
                        }
                    }
                }
                unreachable!()
            }
        }

        return deserializer.deserialize_seq(ResultVisitor {
            deserializer: PhantomData,
            element_seed: PhantomData,
            collector: self.collector,
            initial: self.initial,
            seeds: self.seeds
        });
    }
}

#[stability::unstable(feature = "enable")]
pub struct SerializableNewtype<T>
    where T: Serialize
{
    name: &'static str,
    data: T 
}

impl<T> SerializableNewtype<T>
    where T: Serialize
{
    #[stability::unstable(feature = "enable")]
    pub fn new(name: &'static str, data: T) -> Self {
        Self { name, data }
    }
}

impl<T> Serialize for SerializableNewtype<T>
    where T: Serialize
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        serializer.serialize_newtype_struct(self.name, &self.data)
    }
}

#[stability::unstable(feature = "enable")]
pub struct DeserializeSeedNewtype<'de, S>
    where S: DeserializeSeed<'de>
{
    deserializer: PhantomData<&'de ()>,
    name: &'static str,
    seed: S
}

impl<'de, S> DeserializeSeedNewtype<'de, S>
    where S: DeserializeSeed<'de>
{
    #[stability::unstable(feature = "enable")]
    pub fn new(name: &'static str, seed: S) -> Self {
        Self { deserializer: PhantomData, name, seed }
    }
}

impl<'de, S> DeserializeSeed<'de> for DeserializeSeedNewtype<'de, S>
    where S: DeserializeSeed<'de>
{
    type Value = S::Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where D: Deserializer<'de>
    {
        struct NewtypeStructVisitor<'de, S: DeserializeSeed<'de>> {
            seed: S,
            name: &'static str,
            deserializer: PhantomData<&'de ()>
        }
    
        impl<'de, S: DeserializeSeed<'de>> Visitor<'de> for NewtypeStructVisitor<'de, S> {
            type Value = S::Value;
    
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "a newtype struct named {}", self.name)
            }
    
            fn visit_newtype_struct<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
                where D: Deserializer<'de>
            {
                self.seed.deserialize(deserializer)
            }
        }
    
        return deserializer.deserialize_newtype_struct(self.name, NewtypeStructVisitor { seed: self.seed, name: self.name, deserializer: PhantomData });
    }
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
}

#[cfg(test)]
use crate::integer::{BigIntRing, IntegerRingStore};

#[test]
fn test_serialize() {
    let value = BigIntRing::RING.add(BigIntRing::RING.power_of_two(128), BigIntRing::RING.one());
    let json = serde_json::to_string(&SerializeWithRing::new(&value, BigIntRing::RING)).unwrap();
    assert_eq!("\"340282366920938463463374607431768211457\"", json);
}