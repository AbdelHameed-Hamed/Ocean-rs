#[derive(Debug)]
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
