use core::slice::Iter;
use std::{collections::HashMap, iter::Peekable};

#[derive(Debug, Copy, Clone)]
enum Token<'a> {
    At,
    Identifier(&'a str),
    LCurlyBracket,
    RCurlyBracket,
    LParan,
    RParan,
    Comma,
    Colons,
    LSquareBracket,
    RSquareBracket,
    Number(u32),
    Semicolon,
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
                            tokens.push(Token::Identifier(identifier));
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
                ':' => tokens.push(Token::Colons),
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
        register: (u8, u8),
        type_idx: u32,
    },
    F32xN {
        dimension: u8,
    },
}

struct AST<'a> {
    backing_buffer: Vec<ASTNode<'a>>,
    types: HashMap<&'a str, u32>,
}

fn generate_ast(tokens: Vec<Token>) -> Result<AST, String> {
    let mut res = AST {
        backing_buffer: Vec::new(),
        types: HashMap::new(),
    };

    let mut iter = tokens.iter().peekable();
    while let Some(t) = iter.next() {
        match t {
            Token::At => {
                let name = if let Some(t) = iter.next() {
                    if let Token::Identifier(name) = t {
                        name
                    } else {
                        ""
                    }
                } else {
                    return Err("Incomplete @ scope".to_string());
                };

                let children_begin_idx = if let Some(&t) = iter.next() {
                    if let Token::LCurlyBracket = t {
                        res.backing_buffer.len() as u32
                    } else {
                        return Err(format!("Unexpected token {:?}, expected '{{' instead.", t));
                    }
                } else {
                    return Err("Incomplete @ scope".to_string());
                };

                let idx = res.backing_buffer.len();
                res.backing_buffer.push(ASTNode::None);

                let node = ASTNode::Scope {
                    name,
                    children_begin_idx,
                    children_end_idx: parse_scope(&mut iter, &mut res)?,
                };
                res.backing_buffer[idx] = node;
            }
            Token::Comma => continue,
            _ => {
                return Err(format!(
                    "Unexpected token {:?}, expected '@' or ',' instead.",
                    t
                ))
            }
        }
    }

    return Ok(res);
}

fn parse_scope(iter: &mut Peekable<Iter<Token>>, ast: &mut AST) -> Result<u32, String> {
    while let Some(&t) = iter.next() {
        match t {
            Token::Identifier(name) => {
                if let Some(&t) = iter.next() {
                    if let Token::LParan = t {
                    } else {
                        return Err(format!("Unexpected token {:?}, expected '(' instead.", t));
                    }
                } else {
                    return Err("Incomplete field".to_string());
                }
            }
            _ => {
                return Err(format!("Unexpected token {:?}, expected an identifier.", t));
            }
        }
    }

    return Ok(ast.backing_buffer.len() as u32);
}

fn parse_field<'a>(iter: &mut Peekable<Iter<Token<'a>>>, ast: &mut AST<'a>) -> Result<u32, String> {
    let mut name = "";
    let mut register = (0, 0);
    let mut type_idx = 0;

    while let Some(&t) = iter.next() {
        match t {
            Token::Identifier(ident) => name = ident,
            Token::LParan | Token::Comma | Token::RParan | Token::Colons => continue,
            Token::Number(u32) => (),
            _ => {
                return Err(format!("Unexpected token {:?}, expected an identifier.", t));
            }
        }
    }

    ast.backing_buffer.push(ASTNode::Field {
        name,
        register,
        type_idx,
    });

    unimplemented!()
}
