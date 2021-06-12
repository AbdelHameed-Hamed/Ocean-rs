pub struct PrivateDefer<T>
where
    T: FnMut() -> (),
{
    pub func: T,
}

impl<T> Drop for PrivateDefer<T>
where
    T: FnMut() -> (),
{
    fn drop(&mut self) {
        (self.func)();
    }
}

#[macro_export]
macro_rules! defer {
    ($func:expr) => {
        #[rustfmt::skip]
        let _defer = $crate::defer::PrivateDefer { func: || { $func; } };
    };
}
