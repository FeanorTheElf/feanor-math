
pub mod divisibility;

pub mod pid;

pub mod field;

pub mod ordered;

/// Contains the trait [`integer::IntegerRing`] for rings that represent the ring of integers `Z`.
pub mod integer;

pub mod finite;

/// Contains the trait [`serialization::SerializableElementRing`] for rings whose elements can be
/// serialized by using `serde`.
///
/// It also contains some utilities to simplify this, since it is usually not possible to use
/// `#[derive(Serialize, Deserialize)]` to implement serialization - the reason is that
/// serialization and deserialization usually require access to the ring. Hence, we need to use
/// [`serde::de::DeserializeSeed`], but this is incompatible with `#[derive]`
pub mod serialization;

/// Contains a workaround for specialization.
pub mod specialization;

pub mod reduce_lift;