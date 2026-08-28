use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct TempGguf {
    path: PathBuf,
}

impl TempGguf {
    fn new() -> Self {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "gguf-inspect-test-{}-{id}.gguf",
            std::process::id()
        ));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        fs::write(&path, bytes).expect("create GGUF inspection fixture");

        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempGguf {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[test]
fn inspects_a_memory_mapped_gguf() {
    let fixture = TempGguf::new();
    let output = Command::new(env!("CARGO_BIN_EXE_gguf-inspect"))
        .arg(fixture.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("file bytes: 24"));
    assert!(stdout.contains("version: 3"));
    assert!(stdout.contains("metadata entries: 0"));
    assert!(stdout.contains("tensors: 0"));
    assert!(stdout.contains("data offset: 24"));
}

#[test]
fn rejects_a_file_over_the_explicit_mapping_limit() {
    let fixture = TempGguf::new();
    let output = Command::new(env!("CARGO_BIN_EXE_gguf-inspect"))
        .args(["--max-file-bytes", "23"])
        .arg(fixture.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("file size 24 exceeds mapping limit 23"));
}
