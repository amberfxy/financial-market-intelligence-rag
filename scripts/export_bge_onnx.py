"""Prepare BGE-Large-en-v1.5 for ONNX Runtime inference.

By default downloads the official ONNX export from Hugging Face.
Optionally can export from PyTorch locally with --from-torch.

Usage:
    python scripts/export_bge_onnx.py
    python scripts/export_bge_onnx.py --output-dir models/bge-large-en-v1.5-onnx
    python scripts/export_bge_onnx.py --from-torch --verify
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_OUTPUT = "models/bge-large-en-v1.5-onnx"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "config.json",
]


def download_official_onnx(
    model_name: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_OUTPUT,
) -> Path:
    """Download official onnx/model.onnx + tokenizer into output_dir."""
    from huggingface_hub import hf_hub_download

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading official ONNX from {model_name} (onnx/model.onnx)")
    onnx_src = hf_hub_download(repo_id=model_name, filename="onnx/model.onnx")
    onnx_dst = output_path / "model.onnx"
    shutil.copy2(onnx_src, onnx_dst)
    logger.info(f"Copied ONNX model to {onnx_dst} ({onnx_dst.stat().st_size / 1e6:.1f} MB)")

    for name in TOKENIZER_FILES:
        try:
            src = hf_hub_download(repo_id=model_name, filename=name)
            shutil.copy2(src, output_path / name)
            logger.info(f"Copied {name}")
        except Exception as exc:
            logger.warning(f"Could not download {name}: {exc}")

    return output_path


class _BGEOnnxWrapper:
    """Lazy torch wrapper imported only for --from-torch export."""

    @staticmethod
    def build(model, use_token_type_ids: bool):
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, inner, use_tt):
                super().__init__()
                self.model = inner
                self.use_token_type_ids = use_tt

            def forward(self, input_ids, attention_mask, token_type_ids=None):
                if self.use_token_type_ids:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                else:
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                return outputs.last_hidden_state

        return Wrapper(model, use_token_type_ids)


def export_from_torch(
    model_name: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_OUTPUT,
    opset: int = 14,
    max_length: int = 512,
) -> Path:
    """Export HuggingFace BGE transformer weights to ONNX + tokenizer."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_path = output_path / "model.onnx"

    logger.info(f"Loading PyTorch model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    dummy = tokenizer(
        "Export dummy sentence for ONNX.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=min(32, max_length),
    )
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    use_token_type_ids = "token_type_ids" in dummy
    wrapper = _BGEOnnxWrapper.build(model, use_token_type_ids=use_token_type_ids)

    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
    }

    if use_token_type_ids:
        token_type_ids = dummy["token_type_ids"]
        input_names.append("token_type_ids")
        inputs = (input_ids, attention_mask, token_type_ids)
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "sequence"}
    else:
        inputs = (input_ids, attention_mask)

    logger.info(f"Exporting ONNX model to {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    tokenizer.save_pretrained(str(output_path))
    logger.info(f"Tokenizer saved to {output_path}")
    logger.info(f"Export complete: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def _cls_normalize(last_hidden_state):
    import numpy as np

    cls = last_hidden_state[:, 0, :].astype(np.float32)
    norms = np.linalg.norm(cls, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return cls / norms


def verify_onnx_runtime(output_dir: str) -> None:
    """Smoke-test ONNX Runtime embedder on sample texts."""
    import numpy as np

    from src.embeddings.onnx_embedder import ONNXBGEEmbedder

    samples = [
        "Why did NVDA stock fall after earnings?",
        "Federal Reserve raises interest rates amid inflation concerns.",
        QUERY_INSTRUCTION + "What were the main market trends in 2015?",
    ]

    logger.info("Running ONNX Runtime smoke test...")
    embedder = ONNXBGEEmbedder(model_dir=output_dir)
    embeddings = embedder.embed_texts(samples)

    if embeddings.shape != (len(samples), 1024):
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError(f"Embeddings are not L2-normalized: {norms}")

    cosine = float(np.dot(embeddings[0], embeddings[1]))
    logger.info(
        f"Smoke test passed: shape={embeddings.shape}, "
        f"norms≈1.0, sample cosine(0,1)={cosine:.4f}"
    )


def verify_against_torch(output_dir: str, model_name: str = DEFAULT_MODEL) -> None:
    """Compare PyTorch vs ONNX embeddings (requires loading PyTorch weights)."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoModel, AutoTokenizer

    samples = [
        "Why did NVDA stock fall after earnings?",
        "Federal Reserve raises interest rates amid inflation concerns.",
        QUERY_INSTRUCTION + "What were the main market trends in 2015?",
    ]

    logger.info("Verifying ONNX export against PyTorch...")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    pt_model = AutoModel.from_pretrained(model_name)
    pt_model.eval()

    session = ort.InferenceSession(
        str(Path(output_dir) / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    input_names = {inp.name for inp in session.get_inputs()}

    encoded = tokenizer(
        samples,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        pt_out = pt_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids"),
        )
        pt_emb = _cls_normalize(pt_out.last_hidden_state.cpu().numpy())

    ort_inputs = {
        "input_ids": encoded["input_ids"].numpy().astype(np.int64),
        "attention_mask": encoded["attention_mask"].numpy().astype(np.int64),
    }
    if "token_type_ids" in input_names:
        if "token_type_ids" in encoded:
            ort_inputs["token_type_ids"] = (
                encoded["token_type_ids"].numpy().astype(np.int64)
            )
        else:
            ort_inputs["token_type_ids"] = np.zeros_like(
                ort_inputs["input_ids"], dtype=np.int64
            )

    ort_emb = _cls_normalize(session.run(None, ort_inputs)[0])
    cosine = np.sum(pt_emb * ort_emb, axis=1)

    for i, text in enumerate(samples):
        logger.info(
            f"  sample[{i}] cosine={cosine[i]:.6f} text={text[:60]!r}"
        )

    if float(cosine.min()) < 0.999:
        raise RuntimeError(
            f"ONNX verification failed: min cosine similarity {cosine.min():.6f} < 0.999"
        )
    logger.info("Verification passed: ONNX embeddings match PyTorch (cosine >= 0.999)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare BGE-Large-en-v1.5 ONNX model for ONNX Runtime"
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="HuggingFace model id")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset (torch export only)")
    parser.add_argument(
        "--from-torch",
        action="store_true",
        help="Export from local PyTorch weights instead of downloading official ONNX",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Smoke-test ONNX Runtime embeddings after prepare",
    )
    parser.add_argument(
        "--verify-torch",
        action="store_true",
        help="Compare against PyTorch embeddings (loads full PyTorch model)",
    )
    args = parser.parse_args()

    if args.from_torch:
        export_from_torch(
            model_name=args.model_name,
            output_dir=args.output_dir,
            opset=args.opset,
        )
    else:
        download_official_onnx(
            model_name=args.model_name,
            output_dir=args.output_dir,
        )

    if args.verify:
        verify_onnx_runtime(args.output_dir)

    if args.verify_torch:
        verify_against_torch(args.output_dir, model_name=args.model_name)


if __name__ == "__main__":
    main()
