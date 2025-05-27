use std::ops::Deref;


pub enum MyCow<'a, T> {
    Owned(T),
    #[allow(unused)]
    Mutable(&'a mut T),
    Borrowed(&'a T)
}

impl<'a, T> MyCow<'a, T> {

    pub fn is_owned(&self) -> bool {
        match self {
            Self::Owned(_) => true,
            Self::Mutable(_) => false,
            Self::Borrowed(_) => false
        }
    }

    #[allow(unused)]
    pub fn is_mutable(&self) -> bool {
        match self {
            Self::Owned(_) => true,
            Self::Mutable(_) => true,
            Self::Borrowed(_) => false
        }
    }

    pub fn to_mut_with<F>(&mut self, clone_data: F) -> &mut T
        where F: FnOnce(&T) -> T
    {
        match self {
            Self::Owned(value) => value,
            Self::Mutable(value) => *value,
            Self::Borrowed(value) => {
                *self = MyCow::Owned(clone_data(value));
                match self {
                    Self::Owned(value) => value,
                    _ => unreachable!()
                }
            }
        }
    }

    pub fn to_mut(&mut self) -> &mut T
        where T: Clone
    {
        self.to_mut_with(T::clone)
    }

    #[allow(unused)]
    pub fn is_borrowed(&self) -> bool {
        !self.is_owned()
    }
}

impl<'a, T> Deref for MyCow<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(value) => value,
            Self::Mutable(value) => value,
            Self::Borrowed(value) => *value
        }
    }
}
