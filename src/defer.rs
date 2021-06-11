pub struct PrivateDefer<T>
where
    T: Fn() -> (),
{
    pub func: T,
}

impl<T> Drop for PrivateDefer<T>
where
    T: Fn() -> (),
{
    fn drop(&mut self) {
        (self.func)();
    }
}

#[macro_export(local_inner_macros)]
macro_rules! defer {
    ($func:expr) => {
        #[rustfmt::skip]
        let _defer = PrivateDefer { func: || { $func; } };
    };
}
