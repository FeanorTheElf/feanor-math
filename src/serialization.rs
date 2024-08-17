use std::marker::PhantomData;

use serde::de::{DeserializeSeed, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserializer, Serialize, Serializer};

use crate::ring::*;

#[stability::unstable(feature = "enable")]
pub trait SerializableElementRing: RingBase {

    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de>;

    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer;
}

#[stability::unstable(feature = "enable")]
pub fn deserialize_seq_helper<'de, S, D, C>(deserializer: D, collector: C, base_seed: S) -> Result<(), D::Error>
    where D: Deserializer<'de>,
        C: FnMut(S::Value),
        S: Clone + DeserializeSeed<'de>
{
    struct SeqVisitor<'de, S: Clone + DeserializeSeed<'de>, C: FnMut(S::Value)> {
        base_seed: S,
        collector: C,
        deserializer: PhantomData<&'de ()>
    }

    impl<'de, S: Clone + DeserializeSeed<'de>, C: FnMut(S::Value)> Visitor<'de> for SeqVisitor<'de, S, C> {
        type Value = ();

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(formatter, "a sequence")
        }

        fn visit_seq<A>(mut self, mut seq: A) -> Result<Self::Value, A::Error>
            where A: serde::de::SeqAccess<'de>
        {
            while let Some(el) = seq.next_element_seed(self.base_seed.clone())? {
                (self.collector)(el);
            }
            return Ok(());
        }
    }

    deserializer.deserialize_seq(SeqVisitor {
        deserializer: PhantomData,
        base_seed: base_seed,
        collector: collector
    })
}

#[stability::unstable(feature = "enable")]
pub fn serialize_seq_helper<S, I>(serializer: S, sequence: I) -> Result<S::Ok, S::Error>
    where S: Serializer,
        I: Iterator,
        I::Item: Serialize
{
    let size_hint = sequence.size_hint();
    let mut seq = serializer.serialize_seq(if size_hint.1.is_some() && size_hint.1.unwrap() == size_hint.0 { Some(size_hint.0) } else { None })?;
    for x in sequence {
        seq.serialize_element(&x)?;
    }
    return seq.end();
}

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