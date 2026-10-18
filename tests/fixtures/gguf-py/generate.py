"""Writes small synthetic GGUF files with gguf-py, llama.cpp's reference writer.

The parser's own tests use a writer that shares types with the parser, so a
misunderstanding of the format shared by both would go unnoticed. These files
come from an independent implementation: gguf-py (the `gguf` package on PyPI,
https://github.com/ggml-org/llama.cpp/tree/master/gguf-py), the library that
llama.cpp's convert_hf_to_gguf.py uses to write real models.

Everything in the files is made up: no tensor or vocabulary comes from a real
model. tests/gguf_py.rs checks that gguf-inspect reads back exactly what is
written here.

usage: python -I generate.py OUTPUT_DIR
"""

import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFEndian, GGUFValueType, GGUFWriter


def metadata(w: GGUFWriter) -> None:
    w.add_name("Synthetic gguf-py fixture")
    w.add_quantization_version(2)
    w.add_file_type(7)  # MOSTLY_Q8_0
    w.add_context_length(4096)
    w.add_embedding_length(32)
    w.add_block_count(2)
    w.add_head_count(4)
    w.add_head_count_kv(2)
    w.add_rope_freq_base(10000.0)
    # One entry of every scalar type.
    w.add_uint8("test.u8", 200)
    w.add_int8("test.i8", -100)
    w.add_uint16("test.u16", 60000)
    w.add_int16("test.i16", -30000)
    w.add_uint32("test.u32", 4000000000)
    w.add_int32("test.i32", -2000000000)
    w.add_float32("test.f32", 0.1)
    w.add_bool("test.bool", True)
    w.add_string("test.string", "héllo, wörld")
    w.add_uint64("test.u64", 2**64 - 1)
    w.add_int64("test.i64", -(2**63))
    w.add_float64("test.f64", -1.5e300)
    # Arrays, including typed and nested ones.
    w.add_tokenizer_model("llama")
    w.add_token_list(["<unk>", "<s>", "</s>", "▁a", "▁b"])
    w.add_token_scores([0.0, 0.0, 0.0, -1.0, -2.0])
    w.add_token_types([2, 3, 3, 1, 1])
    w.add_key_value("test.array.u16", [1, 2, 65535], GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT16)
    w.add_key_value("test.array.f64", [0.5, -0.25], GGUFValueType.ARRAY, sub_type=GGUFValueType.FLOAT64)
    w.add_array("test.array.bools", [True, False, True])
    w.add_array("test.array.nested", [[1, 2], [3], ["x", "y"]])


def tensors(w: GGUFWriter) -> None:
    # numpy shapes are outermost first; gguf-py writes them innermost first.
    # Arrays are in native byte order; gguf-py swaps them for big-endian files.
    w.add_tensor("token_embd.weight", np.arange(32 * 5, dtype=np.float32).reshape(5, 32))
    w.add_tensor("blk.0.attn_norm.weight", np.ones(32, dtype=np.float16))
    # A q8_0 tensor of 2 rows x 32 elements, given as raw blocks of 34 bytes.
    w.add_tensor("blk.0.attn_q.weight", np.zeros((2, 34), dtype=np.uint8), raw_dtype=GGMLQuantizationType.Q8_0)
    w.add_tensor("blk.1.ffn_down.weight", np.arange(6, dtype=np.int32).reshape(3, 2))
    w.add_tensor("rope_freqs", np.array(1.5, dtype=np.float64))


def write(path: Path, *, endian: GGUFEndian = GGUFEndian.LITTLE, alignment: int | None = None) -> None:
    w = GGUFWriter(path, "llama", endianess=endian)
    if alignment is not None:
        w.add_custom_alignment(alignment)
    metadata(w)
    tensors(w)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    write(out / "little-endian.gguf")
    write(out / "aligned-64.gguf", alignment=64)
    write(out / "big-endian.gguf", endian=GGUFEndian.BIG)


if __name__ == "__main__":
    main()
