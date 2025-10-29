
pub fn no_error<T>(res: Result<T, !>) -> T {
    match res {
        Ok(res) => res,
        Err(res) => res
    }
}

#[derive(Clone, Copy)]
pub struct TensorProductFunction<F, G>(pub F, pub G);

impl<A, B, F, G> FnOnce<(A, B)> for TensorProductFunction<F, G>
    where F: FnOnce<(A,)>, G: FnOnce<(B,)>
{
    type Output = (F::Output, G::Output);

    extern "rust-call" fn call_once(self, args: (A, B)) -> Self::Output {
        (self.0.call_once((args.0, )), self.1.call_once((args.1, )))
    }
}

impl<A, B, F, G> FnMut<(A, B)> for TensorProductFunction<F, G>
    where F: FnMut<(A,)>, G: FnMut<(B,)>
{
    extern "rust-call" fn call_mut(&mut self, args: (A, B)) -> Self::Output {
        (self.0.call_mut((args.0, )), self.1.call_mut((args.1, )))
    }
}

impl<A, B, F, G> Fn<(A, B)> for TensorProductFunction<F, G>
    where F: Fn<(A,)>, G: Fn<(B,)>
{
    extern "rust-call" fn call(&self, args: (A, B)) -> Self::Output {
        (self.0.call((args.0, )), self.1.call((args.1, )))
    }
}

impl<A, B, F, G> FnOnce<((A, B),)> for TensorProductFunction<F, G>
    where F: FnOnce<(A,)>, G: FnOnce<(B,)>
{
    type Output = (F::Output, G::Output);

    extern "rust-call" fn call_once(self, args: ((A, B),)) -> Self::Output {
        self.call_once(args.0)
    }
}

impl<A, B, F, G> FnMut<((A, B),)> for TensorProductFunction<F, G>
    where F: FnMut<(A,)>, G: FnMut<(B,)>
{
    extern "rust-call" fn call_mut(&mut self, args: ((A, B),)) -> Self::Output {
        self.call_mut(args.0)
    }
}

impl<A, B, F, G> Fn<((A, B),)> for TensorProductFunction<F, G>
    where F: Fn<(A,)>, G: Fn<(B,)>
{
    extern "rust-call" fn call(&self, args: ((A, B),)) -> Self::Output {
        self.call(args.0)
    }
}

#[derive(Clone, Copy)]
pub struct IdentityFunction;

impl<T> FnOnce<(T,)> for IdentityFunction {
    type Output = T;

    extern "rust-call" fn call_once(self, args: (T,)) -> Self::Output {
        args.0
    }
}

impl<T> FnMut<(T,)> for IdentityFunction {
    
    extern "rust-call" fn call_mut(&mut self, args: (T,)) -> Self::Output {
        args.0
    }
}

impl<T> Fn<(T,)> for IdentityFunction {

    extern "rust-call" fn call(&self, args: (T,)) -> Self::Output {
        args.0
    }
}
