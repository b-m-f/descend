use std::io::{Read, Seek, SeekFrom, Write};

pub fn with_captured_output<T>(action: impl FnOnce() -> T) -> (T, String) {
    use std::os::fd::AsRawFd;

    let Ok(mut file) = tempfile::tempfile() else {
        return (action(), String::new());
    };

    let saved_out = unsafe { libc::dup(libc::STDOUT_FILENO) };
    let saved_err = unsafe { libc::dup(libc::STDERR_FILENO) };
    if saved_out < 0 || saved_err < 0 {
        close_saved(saved_out, saved_err);
        return (action(), String::new());
    }

    let target = file.as_raw_fd();
    let redirected = unsafe { libc::dup2(target, libc::STDOUT_FILENO) } >= 0
        && unsafe { libc::dup2(target, libc::STDERR_FILENO) } >= 0;
    if !redirected {
        restore(saved_out, saved_err);
        return (action(), String::new());
    }

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(action));

    let _ = std::io::stdout().flush();
    restore(saved_out, saved_err);

    let mut captured = String::new();
    let _ = file.seek(SeekFrom::Start(0));
    let _ = file.read_to_string(&mut captured);

    match outcome {
        Ok(value) => (value, captured),
        Err(panic) => {
            eprint!("{}", captured);
            std::panic::resume_unwind(panic)
        }
    }
}

fn restore(saved_out: libc::c_int, saved_err: libc::c_int) {
    unsafe {
        libc::dup2(saved_out, libc::STDOUT_FILENO);
        libc::dup2(saved_err, libc::STDERR_FILENO);
    }
    close_saved(saved_out, saved_err);
}

fn close_saved(saved_out: libc::c_int, saved_err: libc::c_int) {
    unsafe {
        if saved_out >= 0 {
            libc::close(saved_out);
        }
        if saved_err >= 0 {
            libc::close(saved_err);
        }
    }
}

pub fn descend_message(captured: &str) -> Option<String> {
    let plain = strip_ansi(captured);

    let messages: Vec<&str> = plain
        .lines()
        .map(str::trim)
        .filter_map(|line| line.strip_prefix("error: "))
        .filter(|message| !message.is_empty())
        .collect();

    if messages.is_empty() {
        eprint!("{}", captured);
        return None;
    }

    Some(messages.join("; "))
}

fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();

    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        for escaped in chars.by_ref() {
            if escaped.is_ascii_alphabetic() {
                break;
            }
        }
    }

    out
}
