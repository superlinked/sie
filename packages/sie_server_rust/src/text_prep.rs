//! SIE text preparation for Rust-backed inference engines.
//!
//! Mirrors the Python adapter rules for `instruction`, `is_query`,
//! `query_template`, and `doc_template` before the request is sent to an
//! engine such as Candle.

pub struct TextPrep<'a> {
    pub instruction: Option<&'a str>,
    pub is_query: bool,
    pub query_template: Option<&'a str>,
    pub doc_template: Option<&'a str>,
}

impl TextPrep<'_> {
    pub fn apply(&self, text: &str) -> String {
        let template = if self.is_query {
            self.query_template
        } else {
            self.doc_template
        };
        if let Some(template) = template.filter(|value| !value.is_empty()) {
            return format_template(template, text, self.instruction.unwrap_or(""));
        }
        if let Some(instruction) = self.instruction.filter(|value| !value.is_empty()) {
            return format!("{instruction} {text}");
        }
        text.to_string()
    }
}

fn format_template(template: &str, text: &str, instruction: &str) -> String {
    let mut out = String::with_capacity(template.len() + text.len() + instruction.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                if chars.peek() == Some(&'{') {
                    out.push('{');
                    chars.next();
                    continue;
                }
                let mut name = String::new();
                let mut closed = false;
                for next in chars.by_ref() {
                    if next == '}' {
                        closed = true;
                        break;
                    }
                    name.push(next);
                }
                if !closed {
                    out.push('{');
                    out.push_str(&name);
                    continue;
                }
                match name.as_str() {
                    "text" => out.push_str(text),
                    "instruction" => out.push_str(instruction),
                    other => {
                        out.push('{');
                        out.push_str(other);
                        out.push('}');
                    }
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    out.push('}');
                    chars.next();
                } else {
                    out.push('}');
                }
            }
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_query_template() {
        let prep = TextPrep {
            instruction: None,
            is_query: true,
            query_template: Some("query: {text}"),
            doc_template: Some("passage: {text}"),
        };
        assert_eq!(prep.apply("cats"), "query: cats");
    }

    #[test]
    fn applies_doc_template() {
        let prep = TextPrep {
            instruction: None,
            is_query: false,
            query_template: Some("query: {text}"),
            doc_template: Some("passage: {text}"),
        };
        assert_eq!(prep.apply("cats"), "passage: cats");
    }

    #[test]
    fn substitutes_instruction() {
        let prep = TextPrep {
            instruction: Some("retrieve relevant passages"),
            is_query: true,
            query_template: Some("Instruct: {instruction}\nQuery: {text}"),
            doc_template: None,
        };
        assert_eq!(
            prep.apply("what is love"),
            "Instruct: retrieve relevant passages\nQuery: what is love"
        );
    }

    #[test]
    fn prepends_bare_instruction_without_template() {
        let prep = TextPrep {
            instruction: Some("Represent this:"),
            is_query: true,
            query_template: None,
            doc_template: None,
        };
        assert_eq!(prep.apply("hello"), "Represent this: hello");
    }

    #[test]
    fn keeps_escaped_braces_literal() {
        let prep = TextPrep {
            instruction: None,
            is_query: true,
            query_template: Some("{{literal}} {text}"),
            doc_template: None,
        };
        assert_eq!(prep.apply("x"), "{literal} x");
    }

    #[test]
    fn preserves_unknown_placeholder() {
        let prep = TextPrep {
            instruction: None,
            is_query: true,
            query_template: Some("{foo}:{text}"),
            doc_template: None,
        };
        assert_eq!(prep.apply("x"), "{foo}:x");
    }
}
