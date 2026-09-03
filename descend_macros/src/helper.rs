/// Internal utility for formatting stdout log statements.
/// 
/// What it does: Suppresses internal debugging blocks unless explicitly configured.
/// Why it is needed: Limits standard console noise escaping during compilation unless explicitly troubleshooting.
macro_rules! debug {
    ($($arg:tt)*) => {
        if std::env::var("DEBUG").is_ok() {
            eprintln!($($arg)*);
        }
    }
}
