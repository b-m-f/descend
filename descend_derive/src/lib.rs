use std::collections::HashSet;

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    Fields, FieldsNamed, Ident, ItemStruct, Token, Type, TypeReference,
};

struct DeriveArguments {
    derive_traits: HashSet<Ident>,
}

impl Parse for DeriveArguments {
    fn parse(input: ParseStream) -> syn::parse::Result<Self> {
        let derive_traits = Punctuated::<Ident, Token![,]>::parse_terminated(input)?;
        Ok(DeriveArguments {
            derive_traits: derive_traits.into_iter().collect(),
        })
    }
}

const IGNORE_ATTRIBUTE_NAME: &str = "span_derive_ignore";

#[proc_macro_attribute]
pub fn span_derive(attributes: TokenStream, input: TokenStream) -> TokenStream {
    let item_struct = parse_macro_input!(input as ItemStruct);
    let arguments = parse_macro_input!(attributes as DeriveArguments);
    if arguments.derive_traits.is_empty() {
        panic!("Need at least one attribute in span_derive!");
    }
    let original_name = &item_struct.ident;
    let original_fields = match &item_struct.fields {
        syn::Fields::Named(fields) => fields,
        _ => unreachable!(),
    };
    let mut stripped_fields = original_fields.clone();
    for field in stripped_fields.named.iter_mut() {
        let mut field_attributes = vec![];
        std::mem::swap(&mut field_attributes, &mut field.attrs);
        field_attributes = field_attributes
            .into_iter()
            .filter(|attribute| !attribute.path.is_ident(IGNORE_ATTRIBUTE_NAME))
            .collect::<Vec<_>>();
        std::mem::swap(&mut field_attributes, &mut field.attrs);
    }
    let mut original = item_struct.clone();
    original.fields = Fields::Named(stripped_fields);
    let mut helper = item_struct.clone();
    helper.generics = syn::parse_str("<'helper>").unwrap();
    let mut helper_fields = FieldsNamed {
        brace_token: original_fields.brace_token.clone(),
        named: Punctuated::new(),
    };
    for mut field in original_fields.named.clone().into_iter() {
        let mut ignored = false;
        for attribute in field.attrs.iter() {
            if attribute.path.is_ident(IGNORE_ATTRIBUTE_NAME) {
                ignored = true;
                break;
            }
        }
        if !ignored {
            let mut reference_type: TypeReference = syn::parse_str("&'helper i64").unwrap();
            reference_type.elem = Box::new(field.ty);
            field.ty = Type::Reference(reference_type);
            helper_fields.named.push(field);
        }
    }
    helper.fields = Fields::Named(helper_fields);
    let helper_name = Ident::new(
        &format!("__{}SpanHelper", original_name.to_string()),
        item_struct.ident.span(),
    );
    helper.ident = helper_name.clone();
    helper.attrs.clear();

    let helper_field_names = helper
        .fields
        .iter()
        .map(|field| field.ident.as_ref().unwrap().clone())
        .collect::<Vec<_>>();

    let derive_arguments = arguments.derive_traits.iter();
    let mut output = quote!(
        #original

        #[derive(#(#derive_arguments),*)]
        #helper

        impl<'a> From<&'a #original_name> for #helper_name<'a> {
            fn from(original: &'a #original_name) -> Self {
                #helper_name {
                    #(#helper_field_names: &original.#helper_field_names),*
                }
            }
        }
    );
    for derive_argument in arguments.derive_traits.iter() {
        match &derive_argument.to_string() as &str {
            "PartialEq" => {
                let implementation = quote!(
                    impl ::core::cmp::PartialEq for #original_name {
                        fn eq(&self, other: &Self) -> bool {
                            let helper: #helper_name = self.into();
                            let helper_other: #helper_name = other.into();
                            helper == helper_other
                        }
                    }
                );
                output.extend(implementation);
            }
            "Eq" => {
                let implementation = quote!(
                    impl ::core::cmp::Eq for #original_name {}
                );
                output.extend(implementation);
            }
            "Hash" => {
                let implementation = quote!(
                    impl ::core::hash::Hash for #original_name {
                        fn hash<H: ::core::hash::Hasher>(&self, state: &mut H) {
                            let helper: #helper_name = self.into();
                            helper.hash(state);
                        }
                    }
                );
                output.extend(implementation);
            }
            _ => panic!("span_derive not implemented for {}", derive_argument),
        }
    }

    TokenStream::from(output)
}
