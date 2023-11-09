use std::any::Any;

pub fn generic_cast<T, U: Any>(val: T) -> Option<U> {

    trait TryCast<U> {
        fn try_cast(self) -> Option<U>;
    }
    impl<T, U> TryCast<U> for T {
        default fn try_cast(self) -> Option<U> { None }
    }
    impl<T> TryCast<T> for T {
        fn try_cast(self) -> Option<T> { Some(self) }
    }

    <T as TryCast<U>>::try_cast(val)
}