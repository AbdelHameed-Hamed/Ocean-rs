use core::slice::Iter;
use std::{collections::HashMap, iter::Peekable};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Token<'a> {
    At,
    Identifier(&'a str),
    LCurlyBracket,
    RCurlyBracket,
    LParan,
    RParan,
    Comma,
    Colon,
    LSquareBracket,
    RSquareBracket,
    Number(u32),
    Semicolon,
    Mut,
    Struct,
}

pub fn tokenize(shader_src: &str) -> Result<Vec<Token>, String> {
    if let Some(start_idx) = shader_src.find('@') {
        let start = unsafe { shader_src.get_unchecked(start_idx..) };
        let mut open_curly_brackets = 0;
        let mut open_parans = 0;
        let mut open_square_brackets = 0;
        let mut iter = start.char_indices().peekable();
        let mut tokens = Vec::<Token>::new();
        let mut started_parsing = false;

        while let Some((i, c)) = iter.next() {
            match c {
                '@' => tokens.push(Token::At),
                'A'..='Z' | 'a'..='z' => {
                    while let Some(&(j, c)) = iter.peek() {
                        if c.is_alphanumeric() == false && c != '_' {
                            let identifier = unsafe { shader_src.get_unchecked(i..j) };
                            if identifier == "mut" {
                                tokens.push(Token::Mut);
                            } else if identifier == "struct" {
                                tokens.push(Token::Struct);
                            } else {
                                tokens.push(Token::Identifier(identifier));
                            }
                            break;
                        }
                        iter.next();
                    }
                }
                '{' => {
                    if started_parsing == false {
                        started_parsing = true;
                    }
                    open_curly_brackets += 1;
                    tokens.push(Token::LCurlyBracket);
                }
                '}' => {
                    open_curly_brackets -= 1;
                    tokens.push(Token::RCurlyBracket);
                }
                '(' => {
                    open_parans += 1;
                    tokens.push(Token::LParan);
                }
                ')' => {
                    open_parans -= 1;
                    tokens.push(Token::RParan);
                }
                ',' => tokens.push(Token::Comma),
                ':' => tokens.push(Token::Colon),
                '[' => {
                    open_square_brackets += 1;
                    tokens.push(Token::LSquareBracket);
                }
                ']' => {
                    open_square_brackets -= 1;
                    tokens.push(Token::RSquareBracket);
                }
                '0'..='9' => {
                    while let Some(&(j, c)) = iter.peek() {
                        if c.is_digit(10) == false {
                            let number = unsafe { shader_src.get_unchecked(i..j) };
                            match number.parse::<u32>() {
                                Ok(n) => tokens.push(Token::Number(n)),
                                Err(parse_error) => return Err(parse_error.to_string()),
                            }
                            break;
                        }
                        iter.next();
                    }
                }
                ';' => tokens.push(Token::Semicolon),
                _ => (),
            }

            if open_curly_brackets == 0 && started_parsing {
                break;
            }
        }

        if open_curly_brackets > 0 || open_parans > 0 || open_square_brackets > 0 {
            return Err("Mismatched closing pairs".to_string());
        }

        return Ok(tokens);
    } else {
        return Ok(Vec::new());
    }
}

#[derive(Debug, Clone, Copy)]
enum PrimitiveType {
    None,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

#[derive(Debug, Clone, Copy)]
struct Structure<'a> {
    name: Option<&'a str>,
    children_begin_idx: usize,
    children_end_idx: usize,
}

#[derive(Debug, Clone, Copy)]
enum BufferType<'a> {
    None,
    PrimitiveType(PrimitiveType),
    Structure(Structure<'a>),
}

#[derive(Debug, Clone, Copy)]
enum FieldType<'a> {
    None,
    PrimitiveType(PrimitiveType),
    Structure(Structure<'a>),
    Buffer {
        read_write: bool,
        underlying_type: BufferType<'a>,
    },
    Texture {
        read_write: bool,
        underlying_type: PrimitiveType,
        underlying_type_count: u8,
        dimension: u8,
    },
}

#[derive(Debug, Clone, Copy)]
enum ASTNode<'a> {
    None,
    Structure(Structure<'a>),
    Field {
        name: &'a str,
        field_type: FieldType<'a>,
        register: Option<(u8, u8)>,
    },
}

pub struct AST<'a> {
    backing_buffer: Vec<ASTNode<'a>>,
    types_map: HashMap<&'a str, (usize, usize)>,
}

pub fn parse(tokens: Vec<Token>) -> Result<AST, String> {
    let mut res = AST {
        backing_buffer: Vec::new(),
        types_map: HashMap::new(),
    };

    let mut iter = tokens.iter().peekable();

    let begin_idx = res.backing_buffer.len();
    res.backing_buffer.push(ASTNode::None);

    consume_token(&mut iter, Token::At)?;

    let mut name = "";
    if let Token::Identifier(ident) = consume_token(&mut iter, Token::Identifier(""))? {
        name = ident;
    }

    consume_token(&mut iter, Token::LCurlyBracket)?;

    parse_structure(&mut iter, &mut res)?;

    consume_token(&mut iter, Token::RCurlyBracket)?;

    let children_end_idx = res.backing_buffer.len();

    res.backing_buffer[begin_idx] = ASTNode::Structure(Structure {
        name: Some(name),
        children_begin_idx: begin_idx + 1,
        children_end_idx,
    });

    return Ok(res);
}

fn parse_structure<'a>(
    iter: &mut Peekable<Iter<Token<'a>>>,
    ast: &mut AST<'a>,
) -> Result<(), String> {
    while let Some(&&token) = iter.peek() {
        if token == Token::RCurlyBracket {
            break;
        }

        parse_field(iter, ast)?;
    }

    return Ok(());
}

fn parse_field<'a>(iter: &mut Peekable<Iter<Token<'a>>>, ast: &mut AST<'a>) -> Result<(), String> {
    let insertion_idx = ast.backing_buffer.len();
    ast.backing_buffer.push(ASTNode::None);

    let mut name: &str = "";

    if let Token::Identifier(ident) = consume_token(iter, Token::Identifier(""))? {
        name = ident;
    }

    consume_token(iter, Token::Colon)?;

    let field_type = parse_field_type(iter, ast)?;

    let register: Option<(u8, u8)>;
    if let Some(Token::LParan) = iter.peek() {
        let mut binding_slot = 0;
        let mut space = 0;

        iter.next();
        if let Token::Number(num) = consume_token(iter, Token::Number(0))? {
            binding_slot = num as u8;
        }

        consume_token(iter, Token::Comma)?;

        if let Token::Number(num) = consume_token(iter, Token::Number(0))? {
            space = num as u8;
        }

        consume_token(iter, Token::RParan)?;

        register = Some((binding_slot, space));
    } else {
        register = None;
    }

    consume_token(iter, Token::Comma)?;

    ast.backing_buffer[insertion_idx as usize] = ASTNode::Field {
        name,
        field_type,
        register,
    };

    return Ok(());
}

fn parse_field_type<'a>(
    iter: &mut Peekable<Iter<Token<'a>>>,
    ast: &mut AST<'a>,
) -> Result<FieldType<'a>, String> {
    let mut mutable = false;
    if let Some(Token::Mut) = iter.peek() {
        mutable = true;
        iter.next();
    }

    if let Some(Token::LSquareBracket) = iter.peek() {
        iter.next();

        let mut underlying_type = BufferType::None;
        if let Token::Identifier(type_name) = consume_token(iter, Token::Identifier(""))? {
            underlying_type = if let Ok(primitive_type) = get_primitive_type(type_name) {
                BufferType::PrimitiveType(primitive_type)
            } else {
                let (children_begin_idx, children_end_idx) = ast.types_map[type_name];
                BufferType::Structure(Structure {
                    name: Some(type_name),
                    children_begin_idx,
                    children_end_idx,
                })
            };
        }

        consume_token(iter, Token::RSquareBracket)?;

        return Ok(FieldType::Buffer {
            read_write: mutable,
            underlying_type,
        });
    } else if let Token::Identifier(type_name) = consume_token(iter, Token::Identifier(""))? {
        match type_name {
            "Tex1D" | "Tex2D" | "Tex3D" => {
                let dimension = type_name.chars().nth(3).unwrap().to_digit(10).unwrap() as u8;
                assert!(dimension >= 1 && dimension <= 3);

                consume_token(iter, Token::LSquareBracket)?;

                let mut underlying_type = PrimitiveType::None;
                if let Token::Identifier(type_name) = consume_token(iter, Token::Identifier(""))? {
                    underlying_type = get_primitive_type(type_name)?;
                }

                consume_token(iter, Token::Semicolon)?;

                let mut underlying_type_count = 0;
                if let Token::Number(num) = consume_token(iter, Token::Number(0))? {
                    assert!(num >= 1 && num <= 4);
                    underlying_type_count = num as u8;
                }

                consume_token(iter, Token::RSquareBracket)?;

                return Ok(FieldType::Texture {
                    read_write: mutable,
                    underlying_type,
                    underlying_type_count,
                    dimension,
                });
            }
            _ => {
                if mutable {
                    return Err(format!("'mut {:?}' makes no sense.", type_name));
                }

                if let Ok(primitive_type) = get_primitive_type(type_name) {
                    return Ok(FieldType::PrimitiveType(primitive_type));
                } else {
                    let (children_begin_idx, children_end_idx) = ast.types_map[type_name];
                    return Ok(FieldType::Structure(Structure {
                        name: Some(type_name),
                        children_begin_idx,
                        children_end_idx,
                    }));
                }
            }
        }
    }

    unreachable!();
}

fn consume_token<'a>(
    iter: &mut Peekable<Iter<Token<'a>>>,
    token: Token<'a>,
) -> Result<Token<'a>, String> {
    if let Some(&t) = iter.next() {
        match t {
            Token::At
            | Token::LCurlyBracket
            | Token::RCurlyBracket
            | Token::LParan
            | Token::RParan
            | Token::Comma
            | Token::Colon
            | Token::LSquareBracket
            | Token::RSquareBracket
            | Token::Semicolon
            | Token::Mut
            | Token::Struct => {
                if token == t {
                    return Ok(t);
                }
            }
            Token::Identifier(_) => {
                if let Token::Identifier(_) = token {
                    return Ok(t);
                }
            }
            Token::Number(_) => {
                if let Token::Number(_) = token {
                    return Ok(t);
                }
            }
        }

        return Err(format!("Expected token {:?}, found {:?}.", token, t));
    } else {
        return Err(format!("Expected token {:?}, found EOF.", token));
    }
}

fn get_primitive_type(type_name: &str) -> Result<PrimitiveType, String> {
    match type_name {
        "i8" => return Ok(PrimitiveType::I8),
        "u8" => return Ok(PrimitiveType::U8),
        "i16" => return Ok(PrimitiveType::I16),
        "u16" => return Ok(PrimitiveType::U16),
        "i32" => return Ok(PrimitiveType::I32),
        "u32" => return Ok(PrimitiveType::U32),
        "i64" => return Ok(PrimitiveType::I64),
        "u64" => return Ok(PrimitiveType::U64),
        "f32" => return Ok(PrimitiveType::F32),
        "f64" => return Ok(PrimitiveType::F64),
        _ => return Err(format!("Unknown primitive type {}", type_name)),
    }
}

pub fn print_ast(ast: &AST) {
    print_ast_internal(ast, 0, 0);
}

fn print_ast_internal(ast: &AST, idx: usize, depth: usize) {
    print!("{:\t<1$}", "", depth);

    let node = &ast.backing_buffer[idx];
    match node {
        &ASTNode::Structure(structure) => {
            print_structure(ast, structure, depth);
        }
        &ASTNode::Field { .. } => println!("{:?}", node),
        _ => unreachable!(),
    }
}

fn print_structure(ast: &AST, structure: Structure, depth: usize) {
    print!("{:\t<1$}", "", depth);
    if let Some(name) = structure.name {
        print!("{}\n", name);
    }

    for i in structure.children_begin_idx..structure.children_end_idx {
        print_ast_internal(ast, i, depth + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_STR: &str = "@Input {
    waves: Tex2D[f32; 4],
    displacement_output_input: mut Tex2D[f32; 4],
    displacement_input_output: mut Tex2D[f32; 4],
    derivatives_output_input: mut Tex2D[f32; 4],
    derivatives_input_output: mut Tex2D[f32; 4],
}";

    #[test]
    fn tokenizer() {
        let tokens = vec![
            Token::At,
            Token::Identifier("Input"),
            Token::LCurlyBracket,
            Token::Identifier("waves"),
            Token::Colon,
            Token::Identifier("Tex2D"),
            Token::LSquareBracket,
            Token::Identifier("f32"),
            Token::Semicolon,
            Token::Number(4),
            Token::RSquareBracket,
            Token::Comma,
            Token::Identifier("displacement_output_input"),
            Token::Colon,
            Token::Mut,
            Token::Identifier("Tex2D"),
            Token::LSquareBracket,
            Token::Identifier("f32"),
            Token::Semicolon,
            Token::Number(4),
            Token::RSquareBracket,
            Token::Comma,
            Token::Identifier("displacement_input_output"),
            Token::Colon,
            Token::Mut,
            Token::Identifier("Tex2D"),
            Token::LSquareBracket,
            Token::Identifier("f32"),
            Token::Semicolon,
            Token::Number(4),
            Token::RSquareBracket,
            Token::Comma,
            Token::Identifier("derivatives_output_input"),
            Token::Colon,
            Token::Mut,
            Token::Identifier("Tex2D"),
            Token::LSquareBracket,
            Token::Identifier("f32"),
            Token::Semicolon,
            Token::Number(4),
            Token::RSquareBracket,
            Token::Comma,
            Token::Identifier("derivatives_input_output"),
            Token::Colon,
            Token::Mut,
            Token::Identifier("Tex2D"),
            Token::LSquareBracket,
            Token::Identifier("f32"),
            Token::Semicolon,
            Token::Number(4),
            Token::RSquareBracket,
            Token::Comma,
            Token::RCurlyBracket,
        ];

        let res = tokenize(TEST_STR).unwrap();
        assert_eq!(res.len(), tokens.len());

        for (i, token) in res.into_iter().enumerate() {
            assert_eq!(token, tokens[i]);
        }
    }
}
