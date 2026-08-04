use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: descendc <file_path>");
        std::process::exit(1);
    }
    match descend::compile(&args[1]) {
        Ok(code) => {
            print!("{}", code);
        }
        Err(e) => {
            eprintln!("Error compiling: {:?}", e);
            std::process::exit(1);
        }
    }
}
