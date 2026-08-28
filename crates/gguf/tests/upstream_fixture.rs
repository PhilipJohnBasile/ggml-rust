use std::fs::File;
use std::path::PathBuf;

use ggml_gguf::{Gguf, GgufHeader, MetadataValue, ScalarValue};

const PROVENANCE_FIXTURE: &[u8] = include_bytes!("fixtures/gguf-py-0.19.0-provenance-v3.gguf");

#[test]
fn parses_current_gguf_py_indexed_provenance_keys() {
    let file = Gguf::from_bytes(PROVENANCE_FIXTURE).unwrap();

    assert_eq!(file.version(), 3);
    assert_eq!(file.tensors().len(), 0);
    assert_eq!(
        file.metadata_value("general.base_model.0.name"),
        Some(&MetadataValue::Scalar(ScalarValue::String("base")))
    );
    assert_eq!(
        file.metadata_value("general.base_model.0.author"),
        Some(&MetadataValue::Scalar(ScalarValue::String("author")))
    );
    assert_eq!(
        file.metadata_value("general.dataset.0.name"),
        Some(&MetadataValue::Scalar(ScalarValue::String("dataset")))
    );
}

#[test]
fn validates_current_gguf_py_fixture_from_file() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gguf-py-0.19.0-provenance-v3.gguf");
    let mut file = File::open(fixture).unwrap();

    let header = GgufHeader::from_reader(&mut file).unwrap();

    assert_eq!(header.version(), 3);
    assert_eq!(header.metadata_count(), 5);
    assert_eq!(header.tensor_count(), 0);
    assert_eq!(header.data_size(), 0);
    assert_eq!(header.data_offset(), header.header_size());
    assert!(header.header_size() <= header.file_size());
}
