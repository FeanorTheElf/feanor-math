
///
/// The absolute error in the expression `exp(2 * pi * i * (x / y))`.
/// 
#[stability::unstable(feature = "enable")]
pub fn root_of_unity_error() -> f64 {
    6. * f64::EPSILON
}

///
/// Trait for operations that perform float-point computations, and thus must
/// care about precision. Currently only used for [`crate::algorithms::fft::FFTAlgorithm`].
/// 
#[stability::unstable(feature = "enable")]
pub trait FFTErrorEstimate {

    ///
    /// This is only true if the table is created with the [`crate::rings::float_complex::Complex64`]-specific creator functions.
    /// Note that this is a worst-case estimate and likely to significantly overestimate the error.
    /// 
    /// This estimates the error from [`super::FFTAlgorithm::unordered_fft()`]. The error during the inverse
    /// FFT is the same, but will be scaled by `1/n`.
    ///
    fn expected_absolute_error(&self, input_bound: f64, input_error: f64) -> f64;
}
