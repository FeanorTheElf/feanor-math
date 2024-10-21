use std::{fmt::Arguments, io::Write, sync::atomic::{AtomicBool, Ordering}};

use atomicbox::AtomicOptionBox;

use crate::unstable_sealed::UnstableSealed;

///
/// Provides an idiomatic way to convert a `Result<T, !>` into `T`, via
/// ```
/// # #![feature(never_type)]
/// # use feanor_math::computation::*;
/// fn some_computation() -> Result<&'static str, !> { Ok("this computation does not fail") }
/// println!("{}", some_computation().unwrap_or_else(no_error));
/// ```
/// 
pub fn no_error<T>(error: !) -> T {
    error
}

///
/// Trait for objects that observe a potentially long-running computation.
/// 
/// This is currently unstable-sealed, since I expect significant additional functionality,
/// potentially including
///  - Early aborts, timeouts
///  - Multithreading
///  - Logging
///  - ...
/// 
/// As a user, this trait should currently be used by passing either [`LogProgress`]
/// or [`DontObserve`] to algorithms.
///  
pub trait ComputationController: Clone + UnstableSealed {

    type Abort: Send;

    ///
    /// Called by algorithms in (more or less) regular time intervals, can provide
    /// e.g. early aborts or tracking progress.
    /// 
    #[stability::unstable(feature = "enable")]
    fn checkpoint(&self, _description: Arguments) -> Result<(), Self::Abort> { 
        Ok(())
    }

    #[stability::unstable(feature = "enable")]
    fn log(&self, _description: Arguments) {}

    ///
    /// Inspired by Rayon, and behaves the same as `join()` there.
    /// Concretely, this function runs both closures, possibly in parallel, and
    /// returns their results.
    /// 
    /// As a convention, if the executor wants to run one of the tasks on
    /// the main thread, it is recommended to choose the second one. Conversely,
    /// if a client intends to call join multiple times to "spawn" more tasks, it
    /// is recommended to again do so in the second task.
    /// 
    #[stability::unstable(feature = "enable")]
    fn join<A, B, RA, RB>(self, oper_a: A, oper_b: B) -> (RA, RB)
        where
            A: FnOnce(Self) -> RA + Send,
            B: FnOnce(Self) -> RB + Send,
            RA: Send,
            RB: Send
    {
        (oper_a(self.clone()), oper_b(self.clone()))
    }
}

///
/// The reason why a (part of a) short-circuiting computation was aborted.
/// 
/// `Finished` means that the computation was aborted, since another part already
/// found a result or aborted. `Abort(e)` means that the controller chose to abort 
/// the computation at a checkpoint, with data `e`.
/// 
pub enum ShortCircuitingComputationAbort<E> {
    Finished,
    Abort(E)
}

///
/// Shared data of a short-circuiting computation.
/// 
pub struct ShortCircuitingComputation<T, Controller>
    where T: Send,
        Controller: ComputationController
{
    finished: AtomicBool,
    abort: AtomicOptionBox<Controller::Abort>,
    result: AtomicOptionBox<T>,
}

///
/// Handle to a short-circuiting computation.
/// 
pub struct ShortCircuitingComputationHandle<'a, T, Controller>
    where T: Send,
        Controller: ComputationController
{
    controller: Controller,
    executor: &'a ShortCircuitingComputation<T, Controller>
}

impl<'a, T, Controller> ShortCircuitingComputationHandle<'a, T, Controller>
    where T: Send,
        Controller: ComputationController
{
    #[stability::unstable(feature = "enable")]
    pub fn controller(&self) -> &Controller {
        &self.controller
    }

    #[stability::unstable(feature = "enable")]
    pub fn checkpoint(&self, description: Arguments) -> Result<(), ShortCircuitingComputationAbort<Controller::Abort>> { 
        if self.executor.finished.load(Ordering::Relaxed) {
            return Err(ShortCircuitingComputationAbort::Finished);
        } else if let Err(e) = self.controller.checkpoint(description) {
            return Err(ShortCircuitingComputationAbort::Abort(e));
        } else {
            return Ok(());
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn log(&self, description: Arguments) {
        self.controller.log(description)
    }

    #[stability::unstable(feature = "enable")]
    pub fn join<A, B>(self, oper_a: A, oper_b: B)
        where
            A: FnOnce(Self) -> Result<Option<T>, ShortCircuitingComputationAbort<Controller::Abort>> + Send,
            B: FnOnce(Self) -> Result<Option<T>, ShortCircuitingComputationAbort<Controller::Abort>> + Send
    {
        let success_fn = |value: T| {
            self.executor.finished.store(true, Ordering::Relaxed);
            self.executor.result.store(Some(Box::new(value)), Ordering::AcqRel);
        };
        let abort_fn = |abort: Controller::Abort| {
            self.executor.finished.store(true, Ordering::Relaxed);
            self.executor.abort.store(Some(Box::new(abort)), Ordering::AcqRel);
        };
        self.controller.join(
            |controller| {
                if self.executor.finished.load(Ordering::Relaxed) {
                    return;
                }
                match oper_a(ShortCircuitingComputationHandle {
                    controller,
                    executor: self.executor
                }) {
                    Ok(Some(result)) => success_fn(result),
                    Err(ShortCircuitingComputationAbort::Abort(abort)) => abort_fn(abort),
                    Err(ShortCircuitingComputationAbort::Finished) => {},
                    Ok(None) => {}
                }
            },
            |controller| {
                if self.executor.finished.load(Ordering::Relaxed) {
                    return;
                }
                match oper_b(ShortCircuitingComputationHandle {
                    controller,
                    executor: self.executor
                }) {
                    Ok(Some(result)) => success_fn(result),
                    Err(ShortCircuitingComputationAbort::Abort(abort)) => abort_fn(abort),
                    Err(ShortCircuitingComputationAbort::Finished) => {},
                    Ok(None) => {}
                }
            }
        );
    }
}

impl<T, Controller> ShortCircuitingComputation<T, Controller>
    where T: Send,
        Controller: ComputationController
{
    #[stability::unstable(feature = "enable")]
    pub fn new() -> Self {
        Self {
            finished: AtomicBool::new(false),
            abort: AtomicOptionBox::none(),
            result: AtomicOptionBox::none()
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn handle<'a>(&'a self, controller: Controller) -> ShortCircuitingComputationHandle<'a, T, Controller> {
        ShortCircuitingComputationHandle {
            controller: controller,
            executor: self
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn finish(self) -> Result<Option<T>, Controller::Abort> {
        if let Some(abort) = self.abort.swap(None, Ordering::AcqRel) {
            return Err(*abort);
        } else if let Some(result) = self.result.swap(None, Ordering::AcqRel) {
            return Ok(Some(*result));
        } else {
            return Ok(None);
        }
    }
}

#[macro_export]
macro_rules! checkpoint {
    ($controller:expr) => {
        ($controller).checkpoint(std::format_args!(""))?
    };
    ($controller:expr, $($args:tt)*) => {
        ($controller).checkpoint(std::format_args!($($args)*))?
    };
}

#[macro_export]
macro_rules! log_progress {
    ($controller:expr, $($args:tt)*) => {
        ($controller).log(std::format_args!($($args)*))
    };
}

#[derive(Clone, Copy)]
pub struct LogProgress;

impl UnstableSealed for LogProgress {}

impl ComputationController for LogProgress {

    type Abort = !;

    fn log(&self, description: Arguments) {
        print!("{}", description);
        std::io::stdout().flush().unwrap();
    }

    fn checkpoint(&self, description: Arguments) -> Result<(), Self::Abort> {
        self.log(description);
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct DontObserve;

impl UnstableSealed for DontObserve {}

impl ComputationController for DontObserve {

    type Abort = !;
}

#[cfg(feature = "parallel")]
mod parallel_controller {

    use super::*;

    #[stability::unstable(feature = "enable")]
    pub struct ExecuteMultithreaded<Rest: ComputationController + Send> {
        rest: Rest
    }

    impl<Rest: ComputationController + Send + Copy> Copy for ExecuteMultithreaded<Rest> {}

    impl<Rest: ComputationController + Send> Clone for ExecuteMultithreaded<Rest> {
        fn clone(&self) -> Self {
            Self { rest: self.rest.clone() }
        }
    }
    
    impl<Rest: ComputationController + Send> UnstableSealed for ExecuteMultithreaded<Rest> {}

    impl<Rest: ComputationController + Send> ComputationController for ExecuteMultithreaded<Rest> {
        type Abort = Rest::Abort;

        #[stability::unstable(feature = "enable")]
        fn checkpoint(&self, description: Arguments) -> Result<(), Self::Abort> { 
            self.rest.checkpoint(description)
        }
    
        #[stability::unstable(feature = "enable")]
        fn log(&self, description: Arguments) {
            self.rest.log(description)
        }
    
        #[stability::unstable(feature = "enable")]
        fn join<A, B, RA, RB>(self, oper_a: A, oper_b: B) -> (RA, RB)
            where
                A: FnOnce(Self) -> RA + Send,
                B: FnOnce(Self) -> RB + Send,
                RA: Send,
                RB: Send
        {
            let self1 = self.clone();
            let self2 = self;
            rayon::join(|| oper_a(self1), || oper_b(self2))
        }
    }

    #[stability::unstable(feature = "enable")]
    #[allow(non_upper_case_globals)]
    pub static RunMultithreadedLogProgress: ExecuteMultithreaded<LogProgress> = ExecuteMultithreaded { rest: LogProgress };
    #[stability::unstable(feature = "enable")]
    #[allow(non_upper_case_globals)]
    pub static RunMultithreaded: ExecuteMultithreaded<DontObserve> = ExecuteMultithreaded { rest: DontObserve };
}

#[cfg(feature = "parallel")]
pub use parallel_controller::*;