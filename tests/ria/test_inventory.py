"""Actual native metadata accounting, with tiny synthetic packages and no bank."""

import copy
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ria.identity import ArtifactError, atomic_json, read_json, seal
from ria.inventory import build_inventory, inventory_request
from test_deployment import deployment_request, measurement_fixture, compose_fixture
from ria.deployment import finalize, _run

ROOT = Path(__file__).resolve().parents[2]


def native_request(tmp_path, role="expert", executor="cpu"):
    request = deployment_request(tmp_path, role, executor)
    request["native_ctl"] = os.environ.get("RIA_CTL_FIXTURE", str(ROOT / "bin/ds4ctl"))
    request["deadline_ms"] = 5000
    return request


def phase_totals(inventory):
    sums = {phase: {"host": 0, "device": 0, "pinned": 0} for phase in
        ("startup", "prefill", "decode", "continuation", "image", "drain")}
    for allocation in inventory["allocations"]:
        for phase in allocation["phases"]:
            sums[phase][allocation["resource"]] += allocation["base_bytes"]
            if allocation["pinned"]:
                sums[phase]["pinned"] += allocation["base_bytes"]
    return sums


@pytest.mark.parametrize("policy", ["sharded", "replicated_experts", "replicated_server_model"])
def test_metadata_inventory_uses_all_phases_and_real_replica_accountant(tmp_path, policy):
    request = native_request(tmp_path)
    request["expert"]["numa_policy"] = policy
    request["expert"]["nodes"].append({"node": 1, "cpus": [1], "workers": 1, "local_bytes": str(16 << 20)})
    request["expert"]["startup_host_bytes"] = str(32 << 20)
    request["planning_request"]["caps"]["host_bytes"] = 32 << 20
    request["planning_request"]["caps"]["numa"].append({"node": 1, "bytes": 16 << 20})
    model = Path(request["environment"]["model_dir"])
    inventory = build_inventory(request, model / "manifest.json", tmp_path / "inventory.json")
    assert inventory == build_inventory(request, model / "manifest.json", tmp_path / "again.json")
    assert len({allocation["id"] for allocation in inventory["allocations"]}) == len(inventory["allocations"])
    assert all(total == {"host": 32 << 20, "device": 0, "pinned": 0} for total in phase_totals(inventory).values())
    assert {allocation["numa_node"] for allocation in inventory["allocations"]} == {0, 1}
    # Every payload file can be absent: inspection authenticates descriptors and
    # computes the future population, and cannot be mistaken for loaded weights.
    for path in (model / "tensors").iterdir():
        path.unlink()
    assert build_inventory(request, model / "manifest.json", tmp_path / "absent-payload.json") == inventory


@pytest.mark.parametrize("mutation", ["metadata_hash", "metadata_budget", "node_budget", "startup_budget", "executor", "profile"])
def test_inventory_failures_preserve_existing_output(tmp_path, mutation):
    request = native_request(tmp_path)
    model = Path(request["environment"]["model_dir"])
    destination = tmp_path / "inventory.json"
    before = build_inventory(request, model / "manifest.json", destination)
    if mutation == "metadata_hash":
        manifest = read_json(model / "manifest.json")
        page = model / manifest["tensor_pages"][0]["path"]
        value = read_json(page)
        value["schema_revision"] = 2
        atomic_json(page, seal(value))
    elif mutation == "metadata_budget":
        request["expert"]["host_runtime_bytes"] = "1"
    elif mutation == "node_budget":
        request["expert"]["nodes"][0]["local_bytes"] = "4096"
    elif mutation == "startup_budget":
        request["expert"]["startup_host_bytes"] = str(8 << 20)
    elif mutation == "executor":
        request["expert"]["device_workspace_bytes"] = "1"
    else:
        request["planning_request"]["profile"] = "fp8"
    with pytest.raises(ArtifactError):
        build_inventory(request, model / "manifest.json", destination)
    assert read_json(destination) == before


def test_finalize_rederives_inventory_and_rejects_omitted_or_resealed_allocations(tmp_path, monkeypatch):
    request = native_request(tmp_path)
    monkeypatch.setattr("ria.host.revalidate_host_report", lambda report: report)
    model = Path(request["environment"]["model_dir"])
    inventory = build_inventory(request, model / "manifest.json", tmp_path / "inventory.json")
    probe, calibration, evidence = measurement_fixture(request, tmp_path)
    output = tmp_path / "final"

    def runner(args, deadline, cwd=None):
        if args[0] == request["native_ctl"]:
            return _run(args, deadline, cwd=cwd)
        from ria.identity import canonical
        return canonical(compose_fixture(request, output)) if args[-2:] == ["--format", "json"] else b""

    changed = copy.deepcopy(inventory)
    changed["allocations"][0]["base_bytes"] -= 1
    with pytest.raises(ArtifactError, match="native population/runtime derivation"):
        finalize(request, probe, seal(changed), calibration, output, runner=runner, **evidence)
    assert not output.exists()
    changed = copy.deepcopy(inventory)
    changed["allocations"][0]["phases"] = ["startup"]
    with pytest.raises(ArtifactError, match="native population/runtime derivation"):
        finalize(request, probe, seal(changed), calibration, output, runner=runner, **evidence)
    assert not output.exists()


def test_inventory_identity_cannot_reuse_another_context_or_runtime(tmp_path):
    request = native_request(tmp_path)
    manifest = read_json(Path(request["environment"]["model_dir"]) / "manifest.json")
    before = inventory_request(request, manifest)
    request["planning_request"]["context_positions"] += 1
    assert inventory_request(request, manifest)["runtime_policy_digest"] != before["runtime_policy_digest"]
    request["expert"]["host_runtime_bytes"] = str(9 << 20)
    assert inventory_request(request, manifest)["runtime_policy_digest"] != before["runtime_policy_digest"]


def test_client_inventory_counts_pinned_reserve_once_and_authenticates_placement(tmp_path):
    from ria.client import client_package
    request = native_request(tmp_path, "client", "cuda")
    server = Path(request["environment"]["model_dir"])
    manifest = read_json(server / "manifest.json")
    model = tmp_path / "client-population"
    client_package(server, manifest["digest"], model, chunk_size=4096)
    planning = request["planning_request"]
    placement = seal({"schema_revision": 1, "logical_model_digest": planning["logical_model_digest"],
        "operator_contract_digest": planning["operator_contract_digest"], "server_layout_digest": manifest["layout_digest"],
        "server_executor": "cuda", "schedule": "full_reference", "shared_placement": "client", "expert_policy": "remote",
        "host_expert_cache_bytes": 0, "device_expert_cache_bytes": 0, "engram_cache_bytes": 0, "local_experts": [],
        "runtime": {"tokenizer_file": "/model/tokenizer.bin", "tokenizer_sha256": "b" * 64,
            "tokenizer_memory_bytes": str(1 << 20), "host_state_bytes": str(1 << 20),
            "device_state_bytes": str(1 << 20), "frontend_host_bytes": str(1 << 20),
            "projection_tile_rows": 64, "state_tile_rows": 64, "max_image_patches": 16}})
    request["placement_plan"] = str(tmp_path / "placement.json")
    atomic_json(request["placement_plan"], placement)
    result = build_inventory(request, model / "manifest.json", tmp_path / "client-inventory.json")
    totals = phase_totals(result)
    assert len(set(tuple(total.items()) for total in totals.values())) == 1
    assert totals["decode"]["pinned"] == planning["caps"]["pinned_bytes"]
    assert totals["decode"]["device"] == 1 << 20
    assert sum(item["base_bytes"] for item in result["allocations"] if item["protected_progress"] and
               item["resource"] == "host") == (3 << 20) + planning["caps"]["pinned_bytes"]
    placement["digest"] = "0" * 64
    atomic_json(request["placement_plan"], placement)
    with pytest.raises(ArtifactError, match="identity"):
        build_inventory(request, model / "manifest.json", tmp_path / "rejected.json")


@pytest.mark.parametrize("mutation", ["type", "unknown", "digest", "context"])
def test_native_plan_validates_generated_inventory_derivation(tmp_path, mutation):
    request = native_request(tmp_path)
    model = Path(request["environment"]["model_dir"])
    inventory = build_inventory(request, model / "manifest.json", tmp_path / "inventory.json")
    probe, calibration, _ = measurement_fixture(request, tmp_path)
    changed = copy.deepcopy(inventory)
    if mutation == "type":
        changed["derivation"] = "unchecked"
    elif mutation == "unknown":
        changed["derivation"]["ignored"] = True
    elif mutation == "digest":
        changed["derivation"]["manifest_digest"] = "bad"
    else:
        changed["derivation"]["context_positions"] += 1
    arguments = [request["native_ctl"], "plan"]
    for option, value in (("request", request["planning_request"]), ("inventory", inventory),
                          ("probe", probe), ("calibration", calibration)):
        path = tmp_path / (option + "-plan.json")
        atomic_json(path, value)
        arguments.extend(["--" + option, str(path)])
    destination = tmp_path / "rejected-plan.json"
    arguments.extend(["--output", str(destination)])
    # The identical complete inputs are admitted before mutating only the
    # derivation. This ensures an unrelated fixture error cannot mask the check.
    _run(arguments, request["deadline_ms"])
    assert read_json(destination)["inventory_digest"] == inventory["digest"]
    destination.unlink()
    atomic_json(tmp_path / "inventory-plan.json", seal(changed))
    with pytest.raises(ArtifactError):
        _run(arguments, request["deadline_ms"])
    assert not destination.exists()
