
#[macro_export]
macro_rules! generate_binding_type {
    ([$(_: $binding_ty:ty = $binding_expr:expr),*]) => {
        ($($binding_ty,)*)
    };
}

#[macro_export]
macro_rules! generate_binding_value {
    ([$(_: $binding_ty:ty = $binding_expr:expr),*]) => {
        ($($binding_expr,)*)
    };
}

#[warn(unused_macros)]
macro_rules! named_closure_type {
    ($name:ident < $({$gen_param:tt $(: $($gen_constraint:tt)*)?}),* > $bindings:tt $params:tt $lambda:expr) => {
        pub struct $name<$($gen_param),*> 
            where $($($gen_param: $($gen_constraint)*,)?)*
        {
            args: generate_binding_type!($bindings)
        }
        
        impl<$($gen_param),*> Clone for $name<$($gen_param),*> 
            where $($($gen_param: $($gen_constraint)*,)?)*
                generate_binding_type!($bindings): Clone
        {
            fn clone(&self) -> Self {
                Self {
                    args: self.args.clone()
                }
            }
        }

        impl<$($gen_param),*> Copy for $name<$($gen_param),*> 
            where $($($gen_param: $($gen_constraint)*,)?)*
                generate_binding_type!($bindings): Copy
        {}

        impl<$($gen_param),*> FnOnce<$params> for $name<$($gen_param),*> 
            where $($($gen_param: $($gen_constraint)*,)?)*
        {
            type Output = i64;

            extern "rust-call" fn call_once(self, args: $params) -> Self::Output
            {
                self.call(args)
            } 
        }
        
        impl<$($gen_param),*> FnMut<$params> for $name<$($gen_param),*> 
            where $($($gen_param: $($gen_constraint)*,)?)*
        {
            extern "rust-call" fn call_mut(&mut self, args: $params) -> Self::Output {
                self.call(args)
            } 
        }

        impl<$($gen_param),*> Fn<$params> for $name<$($gen_param),*>
        {
            extern "rust-call" fn call(&self, args: $params) -> Self::Output
            {
                ($lambda)(args, self.args)
            } 
        }
    };
}

#[warn(unused_macros)]
macro_rules! named_closure {
    ($name:ident < $({$gen_param:tt $(: $($gen_constraint:tt)*)?}),* > $bindings:tt $params:tt $lambda:expr) => {
        {
            named_closure_type!{$name < $({$gen_param $(: $($gen_constraint)*)?}),* > $bindings $params $lambda}

            $name {
                args: generate_binding_value!($bindings)
            }
        }
    }
}

#[test]
fn test_named_closure() {
    assert_eq!(
        vec![3, 4],
        [0, 1].iter().map(
            named_closure!(
                AddConstant < {'a} > [ _: &'a i64 = &3 ] (&i64, ) 
                    |(x,): (&i64,), (constant,): (&i64,)| { *x + *constant }
            )
        ).collect::<Vec<_>>()
    )
}