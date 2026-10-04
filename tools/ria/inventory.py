"""Derive an admission inventory with the serving metadata/NUMA accountant."""

from pathlib import Path
import tempfile

from .identity import ArtifactError, atomic_json, digest, read_json, u64, verify_identity
from .schemas import validate


def inventory_request(request, manifest):
    """Freeze reservations; operators choose caps, never internal tensor sizes.

    Reserves are conservative over every phase. CUDA/library/workspace peaks
    still require measured qualification; this producer executes no probe and
    does not claim that requested reservations are physically available.
    """
    validate("deployment-request", request)
    validate("manifest", manifest)
    verify_identity(manifest)
    planning = request["planning_request"]
    role = planning["role"]
    if manifest["role"] != ("client" if role == "client" else "server") or manifest["profile"] != planning["profile"] or any(
            manifest[field] != planning[field] for field in ("logical_model_digest", "operator_contract_digest")):
        raise ArtifactError("inventory manifest differs from requested role/profile/model/operator")
    caps = planning["caps"]
    if role == "expert":
        expert = request.get("expert")
        if not expert:
            raise ArtifactError("expert inventory requires the explicit expert runtime policy")
        host, device, pinned = (u64(expert[field]) for field in
            ("host_runtime_bytes", "device_workspace_bytes", "pinned_workspace_bytes"))
        policy = {"planning_request": planning, "expert": expert, "network": request["network"]}
    else:
        expert = None
        placement = read_json(request["placement_plan"], max_bytes=256 << 10)
        validate("placement-plan", placement)
        verify_identity(placement)
        if any(placement[field] != planning[field] for field in ("logical_model_digest", "operator_contract_digest")):
            raise ArtifactError("inventory placement identifies another model/operator")
        runtime = placement["runtime"]
        pinned = caps["pinned_bytes"]
        host = sum(u64(runtime[field]) for field in
            ("tokenizer_memory_bytes", "host_state_bytes", "frontend_host_bytes")) + pinned
        device = u64(runtime["device_state_bytes"])
        policy = {"planning_request": planning, "placement": placement, "network": request["network"], "api": request["api"]}
    if not 0 < host <= caps["host_bytes"] or device > caps["device_bytes"] or pinned > caps["pinned_bytes"]:
        raise ArtifactError("runtime inventory reservations exceed chosen caps")
    return {"schema_revision": 1, "role": role, "executor": planning["executor"],
        "profile": planning["profile"], "manifest_digest": manifest["digest"],
        "context_positions": planning["context_positions"], "max_metadata_bytes": host,
        "host_cap": caps["host_bytes"], "device_cap": caps["device_bytes"], "pinned_cap": caps["pinned_bytes"],
        "host_runtime_bytes": str(host), "device_runtime_bytes": str(device), "pinned_runtime_bytes": str(pinned),
        "expert": expert, "runtime_policy_digest": digest(policy)}


def build_inventory(request, manifest_path, output, *, runner=None):
    """Authenticate metadata only; no checkpoint payload/model/GPU is loaded."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    raw = inventory_request(request, read_json(manifest_path, max_bytes=256 << 10))
    if runner is None:
        from .deployment import _run
        runner = _run
    with tempfile.TemporaryDirectory(prefix="ria-inventory-") as temporary:
        source, destination = Path(temporary) / "request.json", Path(temporary) / "inventory.json"
        atomic_json(source, raw)
        runner([request["native_ctl"], "inventory", "--manifest", str(manifest_path),
            "--request", str(source), "--output", str(destination)], request["deadline_ms"])
        result = read_json(destination, max_bytes=65536)
        validate("inventory", result)
        verify_identity(result)
        if result.get("derivation") != {"manifest_digest": raw["manifest_digest"],
                "runtime_policy_digest": raw["runtime_policy_digest"], "request_digest": digest(raw),
                "context_positions": raw["context_positions"]}:
            raise ArtifactError("native inventory provenance differs from its inputs")
        atomic_json(output, result)
        return result
