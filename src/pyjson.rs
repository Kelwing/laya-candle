//! Renders JSON exactly as Python's `json.dumps` does (see ADR-0002).
//!
//! Python's default separators are `", "` and `": "`, keys keep insertion order, and floats use
//! Python's `repr`. With `ensure_ascii=True` every non-ASCII character becomes a `\uXXXX`
//! escape (surrogate pairs above the BMP); with `ensure_ascii=False` it is written as-is.

use serde_json::Value;
use std::fmt::Write;

/// `json.dumps(value, ensure_ascii=ensure_ascii)`.
pub fn dumps(value: &Value, ensure_ascii: bool) -> String {
    let mut out = String::new();
    write_value(&mut out, value, ensure_ascii);
    out
}

fn write_value(out: &mut String, value: &Value, ensure_ascii: bool) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                write!(out, "{i}").unwrap();
            } else if let Some(u) = n.as_u64() {
                write!(out, "{u}").unwrap();
            } else {
                write_float(out, n.as_f64().unwrap_or(f64::NAN));
            }
        }
        Value::String(s) => write_string(out, s, ensure_ascii),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, item, ensure_ascii);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_string(out, k, ensure_ascii);
                out.push_str(": ");
                write_value(out, v, ensure_ascii);
            }
            out.push('}');
        }
    }
}

fn write_string(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c if ensure_ascii && (c as u32) >= 0x7f => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    write!(out, "\\u{:04x}", unit).unwrap();
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `float.__repr__`: shortest round-trip digits, fixed notation when the decimal point
/// lands within (-4, 16], otherwise `d.ddde±XX` with a sign and at least two exponent digits.
pub fn write_float(out: &mut String, f: f64) {
    if f.is_nan() {
        out.push_str("NaN");
        return;
    }
    if f.is_infinite() {
        out.push_str(if f < 0.0 { "-Infinity" } else { "Infinity" });
        return;
    }
    if f.is_sign_negative() {
        out.push('-');
    }
    // `{:e}` gives the shortest round-trip mantissa: "1.5e-5", "1e16", "0e0".
    let sci = format!("{:e}", f.abs());
    let (mantissa, exp) = sci
        .split_once('e')
        .expect("Rust `{:e}` always contains 'e'");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let exp: i32 = exp.parse().expect("Rust `{:e}` exponent is an integer");
    let decpt = exp + 1;
    if -4 < decpt && decpt <= 16 {
        if decpt <= 0 {
            out.push_str("0.");
            for _ in 0..(-decpt) {
                out.push('0');
            }
            out.push_str(&digits);
        } else if decpt as usize >= digits.len() {
            out.push_str(&digits);
            for _ in 0..(decpt as usize - digits.len()) {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..decpt as usize]);
            out.push('.');
            out.push_str(&digits[decpt as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        write!(out, "e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs()).unwrap();
    }
}

/// Python `repr(str)`: single-quoted unless the text contains a single quote and no double
/// quote; backslashes, the quote, and control characters are escaped.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                write!(out, "\\x{:02x}", c as u32).unwrap()
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn float(f: f64) -> String {
        let mut s = String::new();
        write_float(&mut s, f);
        s
    }

    #[test]
    fn floats_match_python_repr() {
        assert_eq!(float(1.0), "1.0");
        assert_eq!(float(0.1), "0.1");
        assert_eq!(float(-0.0), "-0.0");
        assert_eq!(float(0.0), "0.0");
        assert_eq!(float(100.0), "100.0");
        assert_eq!(float(1e15), "1000000000000000.0");
        assert_eq!(float(1e16), "1e+16");
        assert_eq!(float(1.5e300), "1.5e+300");
        assert_eq!(float(0.0001), "0.0001");
        assert_eq!(float(0.00001), "1e-05");
        assert_eq!(float(1.2345e-7), "1.2345e-07");
        assert_eq!(float(123456789012345678.0), "1.2345678901234568e+17");
        assert_eq!(float(std::f64::consts::PI), "3.141592653589793");
        assert_eq!(float(2.5), "2.5");
    }

    #[test]
    fn separators_order_and_escapes() {
        let v = json!({"b": [1, 2.0, true, null], "a": {"x": "q\"\\\n\u{1}é"}});
        assert_eq!(
            dumps(&v, false),
            "{\"b\": [1, 2.0, true, null], \"a\": {\"x\": \"q\\\"\\\\\\n\\u0001é\"}}"
        );
        assert_eq!(dumps(&json!("é😀"), true), "\"\\u00e9\\ud83d\\ude00\"");
        assert_eq!(dumps(&json!([]), false), "[]");
        assert_eq!(dumps(&json!({}), false), "{}");
    }
}
