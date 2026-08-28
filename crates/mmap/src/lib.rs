//! Bounded, read-only file mappings for model inspection and runtime planning.
//!
//! A shared advisory lock is held for the lifetime of each mapping. Advisory
//! locks cannot prevent an uncooperative process from truncating or rewriting
//! the backing file, so constructing a file-backed mapping is explicitly unsafe.

use std::fmt;
use std::fs::File;
use std::io;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

/// A read-only mapping whose backing file remains open and shared-locked.
#[derive(Debug)]
pub struct MappedFile {
    // Field order matters: unmap before closing the locked file.
    map: Mmap,
    _file: File,
}

impl MappedFile {
    /// Opens and maps a regular, non-empty file no larger than `max_bytes`.
    ///
    /// The limit is checked before address space is reserved. The file length
    /// is checked again after mapping to detect cooperative race conditions.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the opened file's contents and length
    /// remain unchanged until the returned mapping is dropped. A shared file
    /// lock is held as a cooperative guard, but it does not enforce this rule
    /// against other processes or writable handles.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero limit, an unsuitable file, a failed shared
    /// lock or mapping, a size larger than the supplied limit or host address
    /// space, or a file whose length changes while it is being mapped.
    #[allow(unsafe_code)]
    pub unsafe fn open(path: impl AsRef<Path>, max_bytes: u64) -> Result<Self, MapError> {
        if max_bytes == 0 {
            return Err(MapError::InvalidLimit);
        }

        let file = File::open(path)?;
        file.try_lock_shared().map_err(io::Error::from)?;

        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(MapError::NotRegularFile);
        }

        let file_len = metadata.len();
        if file_len == 0 {
            return Err(MapError::EmptyFile);
        }
        if file_len > max_bytes {
            return Err(MapError::FileTooLarge {
                size: file_len,
                limit: max_bytes,
            });
        }

        let map_len = usize::try_from(file_len)
            .map_err(|_| MapError::AddressSpaceExceeded { size: file_len })?;
        // SAFETY: this method's caller promises that the file remains immutable
        // for the lifetime of the returned mapping.
        let map = unsafe { map_read_only(&file, map_len) }?;

        let mapped_len = u64::try_from(map.len())
            .map_err(|_| MapError::AddressSpaceExceeded { size: file_len })?;
        if mapped_len != file_len {
            return Err(MapError::MapLengthMismatch {
                expected: file_len,
                actual: mapped_len,
            });
        }

        let current_len = file.metadata()?.len();
        if current_len != file_len {
            return Err(MapError::FileSizeChanged {
                expected: file_len,
                actual: current_len,
            });
        }

        Ok(Self { map, _file: file })
    }

    /// Returns the mapped byte length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns whether the mapping is empty.
    ///
    /// `open` rejects empty files, but this complements `len` for slice-like use.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns all mapped bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.map
    }

    /// Returns a checked byte range using file-sized integer inputs.
    #[must_use]
    pub fn range(&self, offset: u64, byte_len: u64) -> Option<&[u8]> {
        let start = usize::try_from(offset).ok()?;
        let byte_len = usize::try_from(byte_len).ok()?;
        let end = start.checked_add(byte_len)?;
        self.map.get(start..end)
    }
}

impl AsRef<[u8]> for MappedFile {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

/// Failure to create a bounded read-only mapping.
#[derive(Debug)]
#[non_exhaustive]
pub enum MapError {
    Io(io::Error),
    InvalidLimit,
    NotRegularFile,
    EmptyFile,
    FileTooLarge { size: u64, limit: u64 },
    AddressSpaceExceeded { size: u64 },
    MapLengthMismatch { expected: u64, actual: u64 },
    FileSizeChanged { expected: u64, actual: u64 },
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::InvalidLimit => f.write_str("mapping byte limit must be greater than zero"),
            Self::NotRegularFile => f.write_str("mapping source is not a regular file"),
            Self::EmptyFile => f.write_str("cannot memory-map an empty file"),
            Self::FileTooLarge { size, limit } => {
                write!(f, "file size {size} exceeds mapping limit {limit}")
            }
            Self::AddressSpaceExceeded { size } => {
                write!(f, "file size {size} does not fit the host address space")
            }
            Self::MapLengthMismatch { expected, actual } => write!(
                f,
                "mapped length {actual} does not match the expected file length {expected}"
            ),
            Self::FileSizeChanged { expected, actual } => write!(
                f,
                "file length changed from {expected} to {actual} bytes while mapping"
            ),
        }
    }
}

impl std::error::Error for MapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for MapError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[allow(unsafe_code)]
unsafe fn map_read_only(file: &File, len: usize) -> io::Result<Mmap> {
    // SAFETY: the caller upholds memmap2's requirement that the backing file is
    // not modified for the lifetime of the map. This is the only OS mapping
    // boundary in the crate.
    unsafe { MmapOptions::new().len(len).map(file) }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct TempFile {
        path: PathBuf,
    }

    impl TempFile {
        fn new(bytes: &[u8]) -> Self {
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("ggml-mmap-test-{}-{id}.bin", std::process::id()));
            fs::write(&path, bytes).expect("create mapping test file");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    #[allow(unsafe_code)]
    fn open_fixture(path: &Path, max_bytes: u64) -> Result<MappedFile, MapError> {
        // SAFETY: each test owns its fixture and does not mutate it until the
        // returned mapping has been dropped.
        unsafe { MappedFile::open(path, max_bytes) }
    }

    #[test]
    fn maps_file_and_checks_ranges() {
        let temp = TempFile::new(b"GGUFpayload");
        let mapped = open_fixture(temp.path(), 64).unwrap();

        assert_eq!(mapped.len(), 11);
        assert!(!mapped.is_empty());
        assert_eq!(mapped.as_bytes(), b"GGUFpayload");
        assert_eq!(mapped.range(4, 7), Some(&b"payload"[..]));
        assert_eq!(mapped.range(10, 2), None);
        assert_eq!(mapped.range(u64::MAX, 1), None);
    }

    #[test]
    fn rejects_zero_limit_before_opening() {
        let error = open_fixture(Path::new("does-not-need-to-exist"), 0).unwrap_err();
        assert!(matches!(error, MapError::InvalidLimit));
    }

    #[test]
    fn rejects_file_larger_than_limit() {
        let temp = TempFile::new(b"1234");
        let error = open_fixture(temp.path(), 3).unwrap_err();
        assert!(matches!(
            error,
            MapError::FileTooLarge { size: 4, limit: 3 }
        ));
    }

    #[test]
    fn rejects_empty_file() {
        let temp = TempFile::new(&[]);
        let error = open_fixture(temp.path(), 1).unwrap_err();
        assert!(matches!(error, MapError::EmptyFile));
    }
}
