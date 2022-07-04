use core::slice::Iter;
use std::{collections::HashMap, iter::Peekable};

#[derive(Debug, Copy, Clone, PartialEq)]
enum Token<'a> {
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
}

fn tokenize(shader_src: &str) -> Result<Vec<Token>, String> {
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

enum ASTNode<'a> {
    None,
    Scope {
        name: &'a str,
        children_begin_idx: u32,
        children_end_idx: u32,
    },
    Field {
        name: &'a str,
        type_idx: u32,
        register: Option<(u8, u8)>,
    },
    Texture {
        read_write: bool,
        dimension: u8,
        underlying_type: (u32, u8),
    },
    Buffer {
        read_write: bool,
        underlying_type_idx: u32,
    },
}

struct AST<'a> {
    backing_buffer: Vec<ASTNode<'a>>,
    types: HashMap<&'a str, u32>,
}

fn parse(tokens: Vec<Token>) -> Result<AST, String> {
    let mut res = AST {
        backing_buffer: Vec::new(),
        types: HashMap::new(),
    };

    let mut iter = tokens.iter().peekable();

    consume_token(&mut iter, Token::At)?;

    let mut name = "";
    if let Token::Identifier(ident) = consume_token(&mut iter, Token::Identifier(""))? {
        name = ident;
    }

    consume_token(&mut iter, Token::LCurlyBracket)?;

    let children_begin_idx = res.backing_buffer.len() as u32;

    parse_scope(&mut iter, &mut res)?;

    consume_token(&mut iter, Token::RCurlyBracket)?;

    let children_end_idx = res.backing_buffer.len() as u32;

    res.backing_buffer.push(ASTNode::Scope {
        name,
        children_begin_idx,
        children_end_idx,
    });

    return Ok(res);
}

fn parse_scope<'a>(iter: &mut Peekable<Iter<Token<'a>>>, ast: &mut AST<'a>) -> Result<(), String> {
    while let Some(&&token) = iter.peek() {
        if token == Token::RCurlyBracket {
            break;
        }

        parse_field(iter, ast)?;
    }

    return Ok(());
}

fn parse_field<'a>(iter: &mut Peekable<Iter<Token<'a>>>, ast: &mut AST<'a>) -> Result<(), String> {
    let mut name: &str = "";

    if let Token::Identifier(ident) = consume_token(iter, Token::Identifier(""))? {
        name = ident;
    }

    consume_token(iter, Token::Colon)?;

    let type_idx = parse_type(iter, ast)?;

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

    ast.backing_buffer.push(ASTNode::Field {
        name,
        type_idx,
        register,
    });

    consume_token(iter, Token::Comma)?;

    return Ok(());
}

fn parse_type<'a>(iter: &mut Peekable<Iter<Token<'a>>>, ast: &mut AST<'a>) -> Result<u32, String> {
    let mut mutable = false;
    if let Some(Token::Mut) = iter.peek() {
        mutable = true;
        iter.next();
    }

    if let Some(Token::LSquareBracket) = iter.peek() {
        iter.next();

        let mut underlying_type_idx = 0;
        if let Token::Identifier(ident) = consume_token(iter, Token::Identifier(""))? {
            underlying_type_idx = ast.types[ident] as u32;
        }

        consume_token(iter, Token::RSquareBracket)?;

        ast.backing_buffer.push(ASTNode::Buffer {
            read_write: mutable,
            underlying_type_idx,
        });

        return Ok(ast.backing_buffer.len() as u32);
    } else if let Token::Identifier(ident) = consume_token(iter, Token::Identifier(""))? {
        match ident {
            "Tex1D" | "Tex2D" | "Tex3D" => {
                let dimension = ident.chars().nth(3).unwrap().to_digit(10).unwrap() as u8;

                consume_token(iter, Token::LSquareBracket)?;

                let mut underlying_type_idx = 0;
                if let Token::Identifier(type_name) = consume_token(iter, Token::Identifier(""))? {
                    underlying_type_idx = ast.types[type_name];
                }

                consume_token(iter, Token::Semicolon)?;

                let mut underlying_type_count = 0;
                if let Token::Number(num) = consume_token(iter, Token::Number(0))? {
                    underlying_type_count = num as u8;
                }

                consume_token(iter, Token::RSquareBracket)?;

                ast.backing_buffer.push(ASTNode::Texture {
                    read_write: mutable,
                    dimension,
                    underlying_type: (underlying_type_idx, underlying_type_count),
                });

                return Ok(ast.backing_buffer.len() as u32);
            }
            _ => {
                if mutable {
                    return Err(format!("'mut {:?}' makes no sense.", ident));
                }

                return Ok(ast.types[ident] as u32);
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
            | Token::Mut => {
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

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_STR: &str = "@Input {
    waves: Tex2D[f32; 4],
    displacement_output_input: RWTex2D[f32; 4],
    displacement_input_output: RWTex2D[f32; 4],
    derivatives_output_input: RWTex2D[f32; 4],
    derivatives_input_output: RWTex2D[f32; 4],
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
            Token::Identifier("RWTex2D"),
            Token::LSquareBracket,
            Token::Identifier("f32"),
            Token::Semicolon,
            Token::Number(4),
            Token::RSquareBracket,
            Token::Comma,
            Token::Identifier("displacement_input_output"),
            Token::Colon,
            Token::Identifier("RWTex2D"),
            Token::LSquareBracket,
            Token::Identifier("f32"),
            Token::Semicolon,
            Token::Number(4),
            Token::RSquareBracket,
            Token::Comma,
            Token::Identifier("derivatives_output_input"),
            Token::Colon,
            Token::Identifier("RWTex2D"),
            Token::LSquareBracket,
            Token::Identifier("f32"),
            Token::Semicolon,
            Token::Number(4),
            Token::RSquareBracket,
            Token::Comma,
            Token::Identifier("derivatives_input_output"),
            Token::Colon,
            Token::Identifier("RWTex2D"),
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
