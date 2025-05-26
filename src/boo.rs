use std::ops::Deref;


///
/// A "borrowed-or-owned" value of `T`.
/// 
/// This behaves like [`std::borrow::Cow`], except that it works for types
/// that are not cloneable.
/// 
pub enum Boo<'a, T> {
    Owned(T),
    Borrowed(&'a T)
}

impl<'a, T> Boo<'a, T> {

    pub fn is_owned(&self) -> bool {
        match self {
            Self::Owned(_) => true,
            Self::Borrowed(_) => false
        }
    }

    #[allow(unused)]
    pub fn is_borrowed(&self) -> bool {
        !self.is_owned()
    }
}

impl<'a, T> Deref for Boo<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(value) => value,
            Self::Borrowed(value) => *value
        }
    }
}
