"""Regenerate the upstream gguf-py provenance fixture.

Run with:
    uv run --with gguf==0.19.0 python generate_provenance_fixture.py
"""

from pathlib import Path

from gguf import GGUFWriter


OUTPUT = Path(__file__).with_name("gguf-py-0.19.0-provenance-v3.gguf")

writer = GGUFWriter(OUTPUT, "llama")
writer.add_name("fixture")
writer.add_base_model_name(0, "base")
writer.add_base_model_author(0, "author")
writer.add_dataset_name(0, "dataset")
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
