#!/usr/bin/env python3
"""Self-test for hy4_synthetic.py (stdlib unittest)."""
from __future__ import annotations
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hy4_synthetic as syn


class Hy4SyntheticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hy4-synth-test-"))
        a = syn.argparse.Namespace(**dict(syn.DEFAULTS))
        a.out = str(cls.tmp)
        a.verify = None
        if isinstance(a.engram_layers, str):
            a.engram_layers = tuple(int(x) for x in a.engram_layers.split(",") if x)
        syn.generate(a)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_generated_files(self):
        self.assertTrue((self.tmp / "config.json").is_file())
        self.assertTrue((self.tmp / "model.safetensors.index.json").is_file())
        shards = sorted(self.tmp.glob("model-*.safetensors"))
        self.assertEqual(len(shards), 4)

    def test_verify_passes(self):
        self.assertEqual(syn.verify(self.tmp), 0)

    def test_config_shape(self):
        cfg = syn._load_json(self.tmp / "config.json")
        t = cfg["text_config"]
        self.assertEqual(cfg["model_type"], "hy_v4_flash")
        self.assertEqual(t["num_hidden_layers"], 4)
        self.assertEqual(t["hidden_size"], 64)
        self.assertEqual(t["num_attention_heads"], 8)
        self.assertEqual(t["hc_mult"], 4)
        self.assertEqual(t["engram_n_embed_per_ngram"], t["engram_n_head_per_ngram"] * 96)
        self.assertEqual(t["engram_layer_ids"], [2])
        self.assertEqual(cfg["dfly_num_hidden_layers"], 1)
        self.assertEqual(cfg["speculative_config"]["num_speculative_tokens"], 2)

    def test_name_count_and_novelty(self):
        import re
        cfg = syn._load_json(self.tmp / "config.json")
        names = set(syn.derive_names(cfg, syn._load_json(syn.INDEX_FIXTURE)))
        official = set(syn._load_json(syn.INDEX_FIXTURE)["weight_map"])
        self.assertEqual(len(names), syn.expected_count(cfg))
        # Layer numbers are re-mapped, so compare name templates instead.
        def template(n):
            return re.sub(r"[0-9]+", "#", n)
        self.assertTrue({template(n) for n in names} <= {template(n) for n in official})


if __name__ == "__main__":
    unittest.main(verbosity=2)
