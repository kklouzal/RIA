#!/usr/bin/env python3
"""Driver-free request failures, independent-oracle and CPU fixture checks.

No GPU metadata query, CUDA launch, model access or acceptance decision occurs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def request(profile="bf16"):
    result = {"schema_revision": 1, "kind": "native_fixture_request", "profile": profile,
              "executor": "cpu", "runner_executor": "cpu", "role": "server", "gpu_uuid": None, "expert_shape": "ragged",
              "host_budget": "4194304", "device_budget": "0", "pinned_budget": "0",
              "deadline_ms": 30000, "repeats": 2, "warmup": 1,
              "fixture_seed": "17", "relative_floor": .0001}
    for name in ("logical_model_digest", "source_lock_digest", "environment_digest", "build_digest",
                 "policy_digest", "operator_contract_digest", "preregistration_digest"):
        result[name] = hashlib.sha256(name.encode()).hexdigest()
    return result


class Qualify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="ria-qualify-offline-")
        cls.path = Path(cls.directory.name)
        if os.environ.get("RIA_QUALIFY_FIXTURE"):
            cls.binary = Path(os.environ["RIA_QUALIFY_FIXTURE"]).resolve(strict=True)
            return
        objects = []
        for source in ("ria/qualify.c", "ria/expert.c", "ria/common.c", "ria/json.c", "third_party/ryu/ryu/d2s.c"):
            output = cls.path / (Path(source).stem + ".o")
            flags = ["-std=c99", "-pedantic", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"]
            if not source.startswith("third_party/"):
                flags += ["-Wconversion"]
            subprocess.run(["gcc", *flags, "-DRIA_QUALIFY_CPU_ONLY", "-I", str(ROOT),
                            "-I", str(ROOT / "third_party/ryu"), "-c", str(ROOT / source), "-o", str(output)], check=True)
            objects.append(output)
        cls.binary = cls.path / "qualify-cpu"
        subprocess.run(["gcc", *map(str, objects), "-lcrypto", "-lm", "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_request(self, document, validate=False):
        path = self.path / "request.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        arguments = [str(self.binary)] + (["--validate-request"] if validate else []) + [str(path)]
        return subprocess.run(arguments, capture_output=True, text=True, check=False)

    def test_independent_oracle_boundaries(self):
        report = json.loads(subprocess.check_output([self.binary, "--oracle-self-test"], text=True))
        self.assertTrue(report["oracle_self_test"])
        self.assertFalse(report["gpu_executed"])

    def test_profiles_actual_cpu_against_separate_scalar_and_unit_oracles(self):
        for profile in ("bf16", "fp8", "nvfp4"):
            with self.subTest(profile=profile):
                run = self.run_request(request(profile))
                self.assertEqual(run.returncode, 0, run.stderr)
                report = json.loads(run.stdout)
                self.assertFalse(report["qualified"])
                self.assertEqual(report["executor"], "cpu")
                self.assertEqual(report["graph_gpu_fixtures"], "unexecuted")
                self.assertEqual(report["transfer_gpu_fixtures"], "unexecuted")
                self.assertEqual(report["full_model_graph_parity"], "unexecuted")
                self.assertEqual(len(report["cases"]), 2)
                self.assertLessEqual(int(report["owned_host_peak_bytes"]), 4194304)
                self.assertEqual(int(report["device_peak_bytes"]), 0)
                self.assertEqual(int(report["pinned_peak_bytes"]), 0)
                for case in report["cases"]:
                    self.assertEqual(case["rows"], 17)
                    self.assertEqual(case["input_features"], 65)
                    self.assertEqual(case["intermediate_features"], 33)
                    self.assertEqual(case["output_features"], 19)
                    self.assertEqual(case["max_abs_error"], 0)
                    self.assertEqual(case["max_relative_error"], 0)
                    self.assertEqual(case["max_rms_error"], 0)
                    self.assertTrue(case["exact_match"])
                    self.assertTrue(case["repeat_identical"])
                    self.assertEqual(case["actual_digest"], case["oracle_digest"])
                    self.assertEqual(len(case["latency_ns"]), 2)
                    self.assertTrue(all(int(value) > 0 for value in case["latency_ns"]))

    def test_replay_and_seed_identity(self):
        a = json.loads(self.run_request(request("nvfp4")).stdout)
        b = json.loads(self.run_request(request("nvfp4")).stdout)
        self.assertEqual(a["request_digest"], b["request_digest"])
        for left, right in zip(a["cases"], b["cases"], strict=True):
            for field in ("input_digest", "oracle_digest", "actual_digest"):
                self.assertEqual(left[field], right[field])
        changed = request("nvfp4")
        changed["fixture_seed"] = "18"
        c = json.loads(self.run_request(changed).stdout)
        self.assertNotEqual(a["request_digest"], c["request_digest"])
        self.assertNotEqual(a["cases"][0]["input_digest"], c["cases"][0]["input_digest"])

    def test_strict_request_boundaries(self):
        mutations = [("executor", "cpu\0cuda"), ("runner_executor", "unknown"), ("role", "client"),
                     ("profile", "bf16\0fp8"), ("relative_floor", 0),
                     ("deadline_ms", 0), ("deadline_ms", 600001), ("repeats", 33), ("repeats", True),
                     ("warmup", 9), ("gpu_uuid", "GPU-00000000-0000-0000-0000-000000000000"),
                     ("device_budget", "1"), ("pinned_budget", "1"), ("expert_shape", "approximate"),
                     ("environment_digest", "A" * 64), ("extra", 1)]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                document = request()
                document[key] = value
                run = self.run_request(document, validate=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(run.stdout, "")
        document = request()
        del document["policy_digest"]
        self.assertNotEqual(self.run_request(document, validate=True).returncode, 0)

    def test_host_exhaustion_has_no_success_shaped_output(self):
        document = request()
        document["host_budget"] = "1"
        run = self.run_request(document)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(run.stdout, "")
        document["host_budget"] = "50000"
        run = self.run_request(document)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(run.stdout, "")

    def test_cuda_metadata_validation_is_offline_and_execution_fails_driver_free(self):
        document = request()
        document.update(executor="cpu", runner_executor="cuda", role="client", gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
                        device_budget="67108864", pinned_budget="8388608")
        run = self.run_request(document, validate=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertFalse(json.loads(run.stdout)["gpu_executed"])
        run = self.run_request(document)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(run.stdout, "")
        self.assertIn("driver-free", run.stderr)


if __name__ == "__main__":
    unittest.main()
