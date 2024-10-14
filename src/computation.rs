use std::{fmt::Arguments, io::Write};

use crate::unstable_sealed::UnstableSealed;

pub trait ComputationController: Clone + UnstableSealed {

    #[stability::unstable(feature = "enable")]
    fn log(&self, _args: Arguments) {
        unimplemented!("this function is not supposed to have a default impl; unfortunately, we can only apply #[unstable] to functions with default impl")
    }
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

    fn log(&self, args: Arguments) {
        print!("{}", args);
        std::io::stdout().flush().unwrap();
    }
}

pub struct NoOpWrite;

#[derive(Clone)]
pub struct DontObserve;

impl UnstableSealed for DontObserve {}

impl ComputationController for DontObserve {

    fn log(&self, _args: Arguments) {}
}
