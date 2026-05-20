
/// Contains the trait [`divisibility::DivisibilityRing`] for rings that provide information
/// about divisibility of their elements.
pub mod divisibility;
/// Contains the trait [`pid::PrincipalIdealRing`] for rings in whom every ideal is principal.
/// Also contains [`pid::EuclideanRing`], which is the simplest way how a ring can become a
/// principal idea ring.
pub mod pid;
/// Contains the trait [`field::Field`] for rings that are fields.
pub mod field;
/// Contains the trait [`ordered::OrderedRing`] for rings with a total ordering that is compatible
/// with the ring operations.
pub mod ordered;
/// Contains the trait [`integer::IntegerRing`] for rings that represent the ring of integers `Z`.
pub mod integer;
/// Contains a workaround for specialization.
pub mod specialization;
/// Contains the trait [`serialization::SerializableElementRing`] for rings whose elements can be
/// serialized by using `serde`.
///
/// It also contains some utilities to simplify this, since it is usually not possible to use
/// `#[derive(Serialize, Deserialize)]` to implement serialization - the reason is that
/// serialization and deserialization usually require access to the ring. Hence, we need to use
/// [`serde::de::DeserializeSeed`], but this is incompatible with `#[derive]`
pub mod serialization;