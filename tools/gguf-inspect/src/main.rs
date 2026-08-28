use std::env;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::ExitCode;

use ggml_gguf::{Gguf, MetadataValue, ScalarValue};
use ggml_mmap::MappedFile;

const DEFAULT_MAX_FILE_BYTES: u64 = 1_u64 << 40;
const DEFAULT_STRING_PREVIEW_CHARS: usize = 256;

#[derive(Debug, PartialEq, Eq)]
struct Options {
    path: PathBuf,
    max_file_bytes: u64,
    full_output: bool,
}

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Inspect(Options),
    Help(String),
    Version,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("gguf-inspect: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let options = match parse_args(env::args_os())? {
        Command::Inspect(options) => options,
        Command::Help(help) => {
            println!("{help}");
            return Ok(());
        }
        Command::Version => {
            println!("gguf-inspect {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
    };

    let bytes = map_immutable_model(&options.path, options.max_file_bytes)
        .map_err(|error| format!("{}: {error}", options.path.display()))?;
    let file = Gguf::from_bytes(bytes.as_ref())?;
    println!("file bytes: {}", bytes.len());
    println!("version: {}", file.version());
    println!("alignment: {}", file.alignment());
    println!("metadata entries: {}", file.metadata().len());
    println!("tensors: {}", file.tensors().len());
    println!("data offset: {}", file.data_offset());

    for entry in file.metadata() {
        match &entry.value {
            MetadataValue::Scalar(value) => println!(
                "metadata {:?} = {}",
                entry.key,
                render_scalar(*value, options.full_output)
            ),
            MetadataValue::Array(array) => println!(
                "metadata {:?} = array<{:?}>[{}]",
                entry.key,
                array.element_type(),
                array.len()
            ),
        }
    }
    for (index, tensor) in file.tensors().iter().enumerate() {
        let file_range = file
            .tensor_data_range(index)
            .ok_or("parsed tensor range is out of bounds")?;
        println!(
            "tensor {:?} {:?} {} offset={} file_offset={} bytes={}",
            tensor.name,
            tensor.shape(),
            tensor.value_type,
            tensor.offset,
            file_range.start,
            tensor.byte_len
        );
    }
    Ok(())
}

fn render_scalar(value: ScalarValue<'_>, full_output: bool) -> String {
    let ScalarValue::String(value) = value else {
        return format!("{value:?}");
    };
    if full_output {
        return format!("{value:?}");
    }
    let mut characters = value.chars();
    let preview = characters
        .by_ref()
        .take(DEFAULT_STRING_PREVIEW_CHARS)
        .collect::<String>();
    if characters.next().is_none() {
        format!("{preview:?}")
    } else {
        format!("{preview:?}...<truncated; {} bytes total>", value.len())
    }
}

#[allow(unsafe_code)]
fn map_immutable_model(
    path: &std::path::Path,
    max_bytes: u64,
) -> Result<MappedFile, ggml_mmap::MapError> {
    // SAFETY: gguf-inspect documents and requires that the model artifact remain
    // immutable for the command's lifetime. The mapping also holds a cooperative
    // shared lock and retains the opened file handle.
    unsafe { MappedFile::open(path, max_bytes) }
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Command, String> {
    let mut args = args.into_iter();
    let program = args.next().unwrap_or_default();
    let remaining = args.collect::<Vec<_>>();
    if remaining.len() == 1 && matches!(remaining[0].to_str(), Some("-h" | "--help")) {
        return Ok(Command::Help(usage(&program)));
    }
    if remaining.len() == 1 && matches!(remaining[0].to_str(), Some("-V" | "--version")) {
        return Ok(Command::Version);
    }

    let mut path = None;
    let mut max_file_bytes = DEFAULT_MAX_FILE_BYTES;
    let mut max_file_bytes_set = false;
    let mut full_output = false;
    let mut positional_only = false;
    let mut index = 0;
    while index < remaining.len() {
        let argument = &remaining[index];
        if !positional_only && argument == "--" {
            positional_only = true;
        } else if !positional_only && argument == "--full" {
            if full_output {
                return Err(usage(&program));
            }
            full_output = true;
        } else if !positional_only && argument == "--max-file-bytes" {
            if max_file_bytes_set {
                return Err(usage(&program));
            }
            index += 1;
            let limit = remaining.get(index).ok_or_else(|| usage(&program))?;
            max_file_bytes = parse_byte_limit(limit)?;
            max_file_bytes_set = true;
        } else {
            let unknown_flag = !positional_only
                && argument
                    .to_str()
                    .is_some_and(|value| value.starts_with('-'));
            if unknown_flag || path.replace(PathBuf::from(argument)).is_some() {
                return Err(usage(&program));
            }
        }
        index += 1;
    }
    let path = path.ok_or_else(|| usage(&program))?;
    Ok(Command::Inspect(Options {
        path,
        max_file_bytes,
        full_output,
    }))
}

fn parse_byte_limit(value: &OsStr) -> Result<u64, String> {
    let text = value
        .to_str()
        .ok_or("--max-file-bytes must be a decimal integer")?;
    let limit = text
        .parse::<u64>()
        .map_err(|_| "--max-file-bytes must be a decimal integer")?;
    if limit == 0 {
        return Err("--max-file-bytes must be greater than zero".to_owned());
    }
    Ok(limit)
}

fn usage(program: &OsStr) -> String {
    format!(
        "usage: {} [--full] [--max-file-bytes <bytes>] [--] <model.gguf>",
        program.to_string_lossy()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args<'a>(values: &'a [&'a str]) -> impl Iterator<Item = OsString> + 'a {
        values.iter().map(OsString::from)
    }

    #[test]
    fn parses_default_mapping_limit() {
        let command = parse_args(args(&["gguf-inspect", "model.gguf"])).unwrap();
        assert_eq!(
            command,
            Command::Inspect(Options {
                path: PathBuf::from("model.gguf"),
                max_file_bytes: DEFAULT_MAX_FILE_BYTES,
                full_output: false,
            })
        );
    }

    #[test]
    fn parses_explicit_mapping_limit() {
        let command = parse_args(args(&[
            "gguf-inspect",
            "--full",
            "--max-file-bytes",
            "4096",
            "model.gguf",
        ]))
        .unwrap();
        assert_eq!(
            command,
            Command::Inspect(Options {
                path: PathBuf::from("model.gguf"),
                max_file_bytes: 4096,
                full_output: true,
            })
        );
    }

    #[test]
    fn rejects_zero_mapping_limit() {
        let error = parse_args(args(&[
            "gguf-inspect",
            "--max-file-bytes",
            "0",
            "model.gguf",
        ]))
        .unwrap_err();
        assert_eq!(error, "--max-file-bytes must be greater than zero");
    }

    #[test]
    fn rejects_unexpected_arguments_with_usage() {
        let error = parse_args(args(&["gguf-inspect", "one.gguf", "two.gguf"])).unwrap_err();
        assert_eq!(
            error,
            "usage: gguf-inspect [--full] [--max-file-bytes <bytes>] [--] <model.gguf>"
        );
    }

    #[test]
    fn supports_help_and_version() {
        assert!(matches!(
            parse_args(args(&["gguf-inspect", "--help"])).unwrap(),
            Command::Help(_)
        ));
        assert_eq!(
            parse_args(args(&["gguf-inspect", "--version"])).unwrap(),
            Command::Version
        );
    }

    #[test]
    fn escapes_and_truncates_untrusted_strings() {
        let hostile = format!("\u{1b}[31m{}", "x".repeat(300));
        let rendered = render_scalar(ScalarValue::String(&hostile), false);
        assert!(rendered.starts_with(r#""\u{1b}[31m"#));
        assert!(rendered.ends_with(&format!("<truncated; {} bytes total>", hostile.len())));
        assert!(!rendered.contains('\u{1b}'));
    }
}
