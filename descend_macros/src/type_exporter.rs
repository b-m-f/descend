use std::env;
use std::fs;
use proc_macro::TokenStream;

pub fn export_type(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    // basic parsing just to extract struct name and fields
    // We will generate #[repr(C)] input
    // And write the Descend string to temp dir
    
    // For now just return the input
    input
}
