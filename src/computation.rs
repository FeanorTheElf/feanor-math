use std::{fmt::Arguments, io::Write, time::{Duration, Instant}};

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
pub trait ComputationController: Clone + UnstableSealed + Send + Sync {

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
    /// Inspired by Rayon, and behaves the same.
    /// Concretely, this function runs both closures, possibly in parallel, and
    /// returns their results.
    /// 
    #[stability::unstable(feature = "enable")]
    fn join<A, B, RA, RB>(&self, oper_a: A, oper_b: B) -> (RA, RB)
        where
            A: FnOnce() -> RA + Send,
            B: FnOnce() -> RB + Send,
            RA: Send,
            RB: Send
    {
        (oper_a(), oper_b())
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

///
/// We use a wrapper around `print!` instead of just `Stdout`, since
/// this works with output capture in tests.
/// 
pub struct ForwardToPrint;

#[derive(Clone)]
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

#[derive(Clone)]
pub struct DontObserve;

impl UnstableSealed for DontObserve {}

impl ComputationController for DontObserve {

    type Abort = !;
}

#[derive(Clone)]
pub struct Timeout(Instant);

#[derive(Debug)]
pub struct TimeoutError;

impl Timeout {

    pub fn timeout_in(duration: Duration) -> Self {
        Self(Instant::now() + duration)
    }
}

impl UnstableSealed for Timeout {}

impl ComputationController for Timeout {

    type Abort = TimeoutError;

    fn checkpoint(&self, _description: Arguments) -> Result<(), Self::Abort> {
        if Instant::now() > self.0 {
            return Err(TimeoutError);
        } else {
            return Ok(());
        }
    }
}
