use std::any::Any;

pub fn generic_cast<T: Any, U: Any>(val: T) -> Option<U> {
    let mut val_opt = Some(val);
    let val_opt_any: &mut dyn Any = &mut val_opt as &mut dyn Any;
    let result: &mut Option<U> = val_opt_any.downcast_mut::<Option<U>>()?;
    return result.take();
}