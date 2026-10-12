use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Mutex, MutexGuard};

static CAPTURE_LOCK: Mutex<()> = Mutex::new(());

fn hold_capture_lock() -> MutexGuard<'static, ()> {
    CAPTURE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub fn with_captured_output<T>(action: impl FnOnce() -> T) -> (T, String) {
    let _held_for_the_whole_redirect = hold_capture_lock();
    capture(action)
}

#[cfg(unix)]
fn capture<T>(action: impl FnOnce() -> T) -> (T, String) {
    use std::os::fd::AsRawFd;

    let Ok(mut capture_file) = tempfile::tempfile() else {
        return (action(), String::new());
    };

    let _ = std::io::stdout().flush();

    let saved_stdout = unsafe { libc::dup(libc::STDOUT_FILENO) };
    let saved_stderr = unsafe { libc::dup(libc::STDERR_FILENO) };
    if saved_stdout < 0 || saved_stderr < 0 {
        close_saved(saved_stdout, saved_stderr);
        return (action(), String::new());
    }

    let target = capture_file.as_raw_fd();
    let redirected = unsafe { libc::dup2(target, libc::STDOUT_FILENO) } >= 0
        && unsafe { libc::dup2(target, libc::STDERR_FILENO) } >= 0;
    if !redirected {
        restore(saved_stdout, saved_stderr);
        return (action(), String::new());
    }

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(action));

    let _ = std::io::stdout().flush();
    restore(saved_stdout, saved_stderr);

    let mut captured = String::new();
    let _ = capture_file.seek(SeekFrom::Start(0));
    let _ = capture_file.read_to_string(&mut captured);

    match outcome {
        Ok(value) => (value, captured),
        Err(panic) => {
            eprint!("{}", captured);
            std::panic::resume_unwind(panic)
        }
    }
}

#[cfg(not(unix))]
fn capture<T>(action: impl FnOnce() -> T) -> (T, String) {
    (action(), String::new())
}

#[cfg(unix)]
fn restore(saved_stdout: libc::c_int, saved_stderr: libc::c_int) {
    unsafe {
        libc::dup2(saved_stdout, libc::STDOUT_FILENO);
        libc::dup2(saved_stderr, libc::STDERR_FILENO);
    }
    close_saved(saved_stdout, saved_stderr);
}

#[cfg(unix)]
fn close_saved(saved_stdout: libc::c_int, saved_stderr: libc::c_int) {
    unsafe {
        if saved_stdout >= 0 {
            libc::close(saved_stdout);
        }
        if saved_stderr >= 0 {
            libc::close(saved_stderr);
        }
    }
}

pub fn descend_message(captured: &str) -> Option<String> {
    let plain_text = strip_ansi(captured);

    let messages: Vec<&str> = plain_text
        .lines()
        .map(str::trim)
        .filter_map(|line| line.strip_prefix("error: "))
        .filter(|message| !message.is_empty())
        .collect();

    if !messages.is_empty() {
        return Some(messages.join("; "));
    }

    let unprefixed: Vec<&str> = plain_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();

    if unprefixed.is_empty() {
        return None;
    }

    eprint!("{}", captured);
    Some(condense(&unprefixed.join("; ")))
}

fn condense(message: &str) -> String {
    const LIMIT: usize = 300;

    if message.chars().count() <= LIMIT {
        return message.to_string();
    }

    let mut truncated: String = message.chars().take(LIMIT).collect();
    truncated.push_str(" [..] (the Descend compiler printed the full diagnostic above)");
    truncated
}

fn strip_ansi(text: &str) -> String {
    let mut stripped = String::with_capacity(text.len());
    let mut chars = text.chars();

    while let Some(character) = chars.next() {
        if character != '\u{1b}' {
            stripped.push(character);
            continue;
        }
        for escaped in chars.by_ref() {
            if escaped.is_ascii_alphabetic() {
                break;
            }
        }
    }

    stripped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prefixed_diagnostic_is_reported() {
        let captured = "error: cannot borrow\n  |\n  ^^^^\n";
        assert_eq!(descend_message(captured).unwrap(), "cannot borrow");
    }

    #[test]
    fn several_prefixed_diagnostics_are_joined() {
        let captured = "error: first\nnoise\nerror: second\n";
        assert_eq!(descend_message(captured).unwrap(), "first; second");
    }

    #[test]
    fn a_borrow_conflict_without_a_prefix_is_not_dropped() {
        let captured = "Conflicting borrow!!!\nConflictingAccess\n";
        let message = descend_message(captured).expect("the diagnostic must not be lost");
        assert!(message.contains("Conflicting borrow"), "{message}");
        assert!(message.contains("ConflictingAccess"), "{message}");
    }

    #[test]
    fn a_prefixed_diagnostic_wins_over_surrounding_noise() {
        let captured = "Conflicting borrow!!!\nerror: cannot borrow\nconflicting with Place\n";
        assert_eq!(descend_message(captured).unwrap(), "cannot borrow");
    }

    #[test]
    fn colour_escapes_do_not_hide_the_prefix() {
        let captured = "\u{1b}[31merror: \u{1b}[0mcannot borrow\n";
        assert_eq!(descend_message(captured).unwrap(), "cannot borrow");
    }

    #[test]
    fn an_ast_dump_is_cut_short_rather_than_reported_whole() {
        let dump = format!("ConflictingBorrow(PlaceExpr {{ {} }})", "x".repeat(4000));
        let message = descend_message(&dump).expect("the diagnostic must not be lost");
        assert!(message.starts_with("ConflictingBorrow"), "{message}");
        assert!(message.ends_with("full diagnostic above)"), "{message}");
        assert!(
            message.chars().count() < 400,
            "a compile_error! this long is unreadable: {} chars",
            message.chars().count()
        );
    }

    #[test]
    fn a_short_unprefixed_diagnostic_is_not_cut() {
        let message = descend_message("ConflictingAccess\n").unwrap();
        assert_eq!(message, "ConflictingAccess");
    }

    #[test]
    fn nothing_captured_stays_none() {
        assert!(descend_message("").is_none());
        assert!(descend_message("  \n\n \n").is_none());
    }

    #[cfg(unix)]
    fn write_to_stdout_descriptor(text: &str) {
        unsafe {
            libc::write(
                libc::STDOUT_FILENO,
                text.as_ptr().cast::<libc::c_void>(),
                text.len(),
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn concurrent_captures_do_not_swallow_each_other() {
        const THREADS: usize = 8;

        let handles: Vec<_> = (0..THREADS)
            .map(|index| {
                std::thread::spawn(move || {
                    let marker = format!("marker for thread {index}");
                    let (_, captured) = with_captured_output(|| {
                        write_to_stdout_descriptor(&marker);
                        for _ in 0..500 {
                            write_to_stdout_descriptor(".");
                        }
                    });
                    (marker, captured)
                })
            })
            .collect();

        for handle in handles {
            let (marker, captured) = handle.join().expect("a capture must not panic");
            assert!(
                captured.contains(&marker),
                "a capture lost its own output; it held {} bytes",
                captured.len()
            );
            for other in 0..THREADS {
                let other_marker = format!("marker for thread {other}");
                if other_marker != marker {
                    assert!(
                        !captured.contains(&other_marker),
                        "captures interleaved: {captured}"
                    );
                }
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_panicking_action_leaves_stdout_redirected_back() {
        let outcome =
            std::panic::catch_unwind(|| with_captured_output(|| panic!("the action failed")));
        assert!(outcome.is_err(), "the panic must be resumed");

        let (_, captured) = with_captured_output(|| {
            write_to_stdout_descriptor("after the panic");
        });
        assert_eq!(captured, "after the panic");
    }
}
