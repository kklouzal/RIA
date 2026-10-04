"""Owned setup artifacts; preparation and admission keep their existing contracts.

Callers serialize setup operations. Workspaces are private to the setup UID;
only completed model files are made readable by runtime group10001. Provisioned
inputs are never chmod/chown/overwrite targets; source setup can add only the
three fixed, pinned native metadata originals. Facts contain identities and paths, never
credential bytes or fabricated resource/qualification measurements.
"""

import copy
import os
from pathlib import Path
import secrets
import stat

from .client import client_package
from .identity import (ArtifactError, DIGEST, _open_directory, atomic_output,
                       canonical, digest, hash_file, open_regular, read_json,
                       read_verified_bytes, seal, verify_identity, within)
from .preparation import estimate, prepare, verify_package
from .schemas import validate
from .target import MODEL, SOURCE_REVISION, _locked_metadata, create_recipe

RUNTIME_GID = 10001
RUNTIME_UID = 10001
SOURCE_LOCK_FILE = Path(__file__).resolve().parents[2] / "locks/source-lock.json"
NATIVE_METADATA = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def private_directory(path):
    """Create only the named directory beneath an existing link-free parent."""
    path = Path(path)
    if not path.is_absolute():
        raise ArtifactError("setup directories require absolute paths")
    parent = _open_directory(path.parent)
    try:
        try:
            os.mkdir(path.name, 0o700, dir_fd=parent)
            os.fsync(parent)
        except FileExistsError:
            pass
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                         dir_fd=parent)
        except OSError as exc:
            raise ArtifactError("setup directory is not an accessible regular directory") from exc
        try:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise ArtifactError("setup directory must be owned by the setup UID with mode0700")
        finally:
            os.close(fd)
    finally:
        os.close(parent)
    return path


def _owned_area(workspace, name):
    root = private_directory(workspace)
    return private_directory(root / name)


def _publish(path, document):
    with atomic_output(path, mode=0o600, immutable=True) as stream:
        stream.write(canonical(document) + b"\n")
    return document


def _directory(path):
    path = Path(path)
    if not path.is_absolute():
        raise ArtifactError("provisioned artifact root must be absolute")
    fd = _open_directory(path)
    os.close(fd)
    return path


def _source_metadata_matches(directory, name, data):
    """Observe a stable exact source leaf without following or mutating it."""
    def snapshot(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=directory)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ArtifactError(f"unsafe existing native source metadata: {name}") from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ArtifactError(f"unsafe existing native source metadata: {name}; regular single-link files are required")
        if info.st_size != len(data):
            raise ArtifactError(f"native source metadata differs from the pinned original: {name}; explicitly preserve/replace it before retry")
        actual = bytearray()
        while len(actual) < len(data):
            block = os.read(fd, min(len(data) - len(actual), 1 << 20))
            if not block:
                break
            actual.extend(block)
        if (actual != data or os.read(fd, 1) or snapshot(os.fstat(fd)) != snapshot(info) or
                snapshot(os.stat(name, dir_fd=directory, follow_symlinks=False)) != snapshot(info)):
            raise ArtifactError(f"native source metadata differs from the pinned original: {name}; explicitly preserve/replace it before retry")
        return True
    finally:
        os.close(fd)


def _publish_source_metadata(directory, name, data, *, staging):
    """Publish one fixed input through the authorized held source inode."""
    temporary, fd, created, primary = ".ria-" + secrets.token_hex(16), None, False, None
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=staging)
        created = True
        os.fchmod(fd, 0o600)
        pending = memoryview(data)
        while pending:
            count = os.write(fd, pending)
            if not count:
                raise ArtifactError(f"native source metadata write made no progress: {name}")
            pending = pending[count:]
        os.fsync(fd)
        closed, fd = fd, None
        os.close(closed)
        try:
            os.link(temporary, name, src_dir_fd=staging, dst_dir_fd=directory, follow_symlinks=False)
        except FileExistsError as error:
            if not _source_metadata_matches(directory, name, data):
                raise ArtifactError(f"native source metadata publication changed: {name}") from error
        os.unlink(temporary, dir_fd=staging)
        created = False
        os.fsync(staging)
        os.fsync(directory)
    except BaseException as error:
        primary = error
        raise
    finally:
        failures = []
        if fd is not None:
            try:
                os.close(fd)
            except OSError as error:
                failures.append(error)
        if created:
            try:
                os.unlink(temporary, dir_fd=staging)
            except OSError as error:
                failures.append(error)
        if failures:
            message = "; ".join(str(error) for error in failures)
            if primary is not None:
                primary.add_note("owned native metadata cleanup failed: " + message)
            else:
                raise ArtifactError("owned native metadata cleanup failed: " + message) from failures[0]


def provision_source_metadata(source):
    """Add missing pinned native metadata to the explicitly writable source.

    Authenticate the packaged originals and preflight all existing names before
    writing. Existing inputs are preserved byte-for-byte and are never replaced,
    chmodded or followed. A conflicting NVIDIA tokenizer configuration remains
    an explicit operator replacement decision, not an automatic overwrite.
    """
    source = _directory(source)
    lock = read_json(SOURCE_LOCK_FILE)
    verify_identity(lock)
    originals = []
    for name in NATIVE_METADATA:
        entries = [entry for entry in lock["references"] if entry.get("repository") == MODEL and
                   entry.get("revision") == SOURCE_REVISION and entry.get("path") == name]
        expected_path = f"locks/metadata/{MODEL}/{SOURCE_REVISION}/{name}"
        if len(entries) != 1 or entries[0].get("local_path") != expected_path or (
                type(entries[0].get("bytes")) is not int or not 0 < entries[0]["bytes"] <= 16 << 20):
            raise ArtifactError(f"pinned native source metadata reference is invalid: {name}")
        entry = entries[0]
        data = read_verified_bytes(_locked_metadata(MODEL, SOURCE_REVISION, name),
                                   expected_sha256=entry["sha256"], max_bytes=16 << 20)
        if len(data) != entry["bytes"]:
            raise ArtifactError(f"pinned native source metadata size differs: {name}")
        originals.append((name, data, entry["sha256"]))
    directory, staging = _open_directory(source), None
    try:
        missing = [(name, data) for name, data, _ in originals if not _source_metadata_matches(directory, name, data)]
        if missing:
            # Operator-provisioned source roots need not be setup-owned. Keep
            # temporary inodes in the already-authorized private recipe area so
            # another source writer cannot replace them before publication.
            try:
                os.mkdir(".ria-recipes", 0o700, dir_fd=directory)
                os.fsync(directory)
            except FileExistsError:
                pass
            staging = os.open(".ria-recipes", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
            info = os.fstat(staging)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise ArtifactError("native source metadata staging requires setup-owned private .ria-recipes")
        for name, data in missing:
            _publish_source_metadata(directory, name, data, staging=staging)
    finally:
        if staging is not None:
            os.close(staging)
        os.close(directory)
    return [{"path": name, "sha256": sha256} for name, _, sha256 in originals]


def _marked_directory(area, name, identity, *, modes=(0o700, 0o750), runtime_owner=False):
    """Reuse only an inode created by this helper, never claim an existing tree."""
    parent = _open_directory(area)
    marker = Path(area) / (name + "-owner.json")
    try:
        created = False
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
            os.fsync(parent)
            created = True
        except FileExistsError:
            pass
        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        try:
            info = os.fstat(child)
            owners = (os.geteuid(), RUNTIME_UID) if runtime_owner else (os.geteuid(),)
            if info.st_uid not in owners or stat.S_IMODE(info.st_mode) not in modes:
                raise ArtifactError("setup artifact directory has unsafe ownership/mode")
            expected = seal({"schema_revision": 1, "name": name, "input_digest": identity,
                             "device": str(info.st_dev), "inode": str(info.st_ino)})
            if created:
                _publish(marker, expected)
            else:
                try:
                    value = read_json(marker)
                except (ArtifactError, OSError) as exc:
                    raise ArtifactError("setup artifact directory lacks its owned-inode marker") from exc
                verify_identity(value)
                if value != expected:
                    raise ArtifactError("setup artifact directory ownership/input marker differs")
        finally:
            os.close(child)
    finally:
        os.close(parent)
    return Path(area) / name


def runtime_readable(root):
    """Change only an owned completed tree, through held directory descriptors.

    Preparation checkpoints remain private. The caller must keep the workspace
    private and serialize publication so no peer can substitute an owned inode.
    Hard-linked regular files are rejected before their modes can affect peers.
    """
    fd = _open_directory(root)
    root_info = os.fstat(fd)
    if root_info.st_uid != os.geteuid() or stat.S_IMODE(root_info.st_mode) not in (0o700, 0o750):
        os.close(fd)
        raise ArtifactError("runtime permission root must be setup-owned with mode0700/0750")
    def visit(directory, private=False, *, change=False):
        info = os.fstat(directory)
        if info.st_uid != os.geteuid():
            raise ArtifactError("runtime permission target is not setup-owned")
        for name in os.listdir(directory):
            entry = os.stat(name, dir_fd=directory, follow_symlinks=False)
            hidden = private or name.startswith(".prepare")
            if entry.st_uid != os.geteuid() or not (stat.S_ISDIR(entry.st_mode) or
                    (stat.S_ISREG(entry.st_mode) and entry.st_nlink == 1)):
                raise ArtifactError("owned model tree contains an unsafe permission target")
            child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK |
                            (os.O_DIRECTORY if stat.S_ISDIR(entry.st_mode) else 0), dir_fd=directory)
            try:
                opened = os.fstat(child)
                if (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino):
                    raise ArtifactError("model permission target changed during inspection")
                if stat.S_ISDIR(opened.st_mode):
                    visit(child, hidden, change=change)
                elif change:
                    if not hidden and opened.st_gid != RUNTIME_GID:
                        os.fchown(child, -1, RUNTIME_GID)
                    os.fchmod(child, 0o600 if hidden else 0o640)
            finally:
                os.close(child)
        if change:
            if not private and info.st_gid != RUNTIME_GID:
                os.fchown(directory, -1, RUNTIME_GID)
            os.fchmod(directory, 0o700 if private else 0o750)
    try:
        # Inspect the complete tree before mutating any inode. The controller
        # alone can write the parent throughout both traversals.
        visit(fd)
        visit(fd, change=True)
    finally:
        os.close(fd)
    return str(root)


def runtime_reports(root):
    """Publish a new empty reports directory for exclusive runtime writes."""
    fd = _open_directory(root)
    try:
        info = os.fstat(fd)
        if info.st_uid == RUNTIME_UID and stat.S_IMODE(info.st_mode) == 0o700:
            return str(root)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700 or os.listdir(fd):
            raise ArtifactError("runtime reports must be a new empty setup-owned0700 directory")
        os.fchown(fd, RUNTIME_UID, RUNTIME_GID)
    finally:
        os.close(fd)
    return str(root)


def runtime_secret(path):
    """Make a setup-owned single regular credential private to runtimeUID10001."""
    with open_regular(path) as stream:
        info = os.fstat(stream.fileno())
        if info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid not in (os.geteuid(), RUNTIME_UID):
            raise ArtifactError("runtime secret must be an owned single-inode0600 regular file")
        if info.st_uid != RUNTIME_UID or info.st_gid != RUNTIME_GID:
            os.fchown(stream.fileno(), RUNTIME_UID, RUNTIME_GID)
    return str(path)


def runtime_secret_directory(workspace):
    """Create/reuse the controller-owned secret parent readable by runtime group."""
    root = _marked_directory(private_directory(workspace), "secrets", digest({"purpose": "runtime_api_secrets"}))
    fd = _open_directory(root)
    try:
        if os.fstat(fd).st_gid != RUNTIME_GID:
            os.fchown(fd, -1, RUNTIME_GID)
        os.fchmod(fd, 0o750)
    finally:
        os.close(fd)
    return str(root)


def runtime_report_directory(workspace):
    """Create/reuse a marked reports inode whose contents belong to the runtime."""
    root = _marked_directory(private_directory(workspace), "reports", digest({"purpose": "runtime_reports"}),
                             modes=(0o700,), runtime_owner=True)
    return runtime_reports(root)


def _tokenizer(root, manifest):
    matches = []
    for reference in manifest["metadata"]:
        wrapper = read_json(within(root, reference["path"]))
        verify_identity(wrapper, reference["digest"])
        if set(wrapper) == {"schema_revision", "path", "source_path", "sha256", "length", "digest"} and (
                isinstance(wrapper["source_path"], str) and Path(wrapper["source_path"]).name == "tokenizer.json"):
            matches.append(wrapper)
    if len(matches) != 1 or matches[0]["sha256"] != manifest["tokenizer_digest"]:
        raise ArtifactError("prepared package must bind one exact native tokenizer")
    relative = matches[0]["path"]
    path = within(root, relative)
    if hash_file(path) != manifest["tokenizer_digest"]:
        raise ArtifactError("prepared tokenizer hash differs from its manifest")
    value = read_json(path, max_bytes=16 << 20)
    model = value.get("model")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    added = value.get("added_tokens")
    if not isinstance(vocab, dict) or not vocab or not isinstance(added, list):
        raise ArtifactError("native tokenizer vocabulary metadata is missing")
    # ByteLevel BPE glyphs decode to one byte each. Added tokens retain their
    # literal UTF8 bytes, exactly as ria/tokenizer.c stores decoded_length.
    alphabet = {chr(b) for b in (*range(33, 127), *range(161, 173), *range(174, 256))}
    alphabet.update(chr(256 + i) for i in range(256 - len(alphabet)))
    tokens = {}
    for text, token in vocab.items():
        if not text or any(character not in alphabet for character in text) or type(token) is not int or not 0 <= token < 129280 or token in tokens:
            raise ArtifactError("native tokenizer vocabulary identity/alphabet is invalid")
        tokens[token] = (text, len(text))
    seen_added = set()
    for item in added:
        if not isinstance(item, dict) or type(item.get("id")) is not int or not 0 <= item["id"] < 129280 or not isinstance(item.get("content"), str) or not item["content"] or item["id"] in seen_added:
            raise ArtifactError("native added-token metadata is invalid")
        token, text = item["id"], item["content"]
        if token in tokens and tokens[token][0] != text:
            raise ArtifactError("native added-token identity conflicts with vocabulary")
        seen_added.add(token)
        tokens[token] = (text, len(text.encode("utf-8")))
    return {"tokenizer_file": "/model/" + relative, "tokenizer_sha256": manifest["tokenizer_digest"],
            "max_token_bytes": max(length for _, length in tokens.values())}


def _identities(server, client):
    for field in ("profile", "logical_model_digest", "operator_contract_digest", "encoding_digest", "tokenizer_digest"):
        if server[field] != client[field]:
            raise ArtifactError("server and compact client model identities differ")
    if server["role"] != "server" or client["role"] != "client":
        raise ArtifactError("setup package role mismatch")


def verify_model_facts(facts):
    """Check the inner manifests as well as the outer authenticated fact record."""
    if not isinstance(facts, dict):
        raise ArtifactError("setup model facts must be an object")
    verify_identity(facts)
    required = {"schema_revision", "profile", "server_manifest", "client_manifest", "selected_tensors",
                "server_manifest_digest", "client_manifest_digest", "server_layout_digest", "client_layout_digest",
                "logical_model_digest", "operator_contract_digest", "encoding_digest", "tokenizer_file",
                "tokenizer_sha256", "max_token_bytes", "digest"}
    if not required <= facts.keys() or type(facts["schema_revision"]) is not int or facts["schema_revision"] != 1:
        raise ArtifactError("setup model facts fields are missing")
    for field in required:
        if field.endswith("digest") or field == "tokenizer_sha256":
            if not isinstance(facts[field], str) or not DIGEST.fullmatch(facts[field]):
                raise ArtifactError("setup model facts digest is invalid")
    if not isinstance(facts["tokenizer_file"], str) or not facts["tokenizer_file"].startswith("/model/") or (
            type(facts["max_token_bytes"]) is not int or not 1 <= facts["max_token_bytes"] <= 16 << 20):
        raise ArtifactError("setup model facts tokenizer metadata is invalid")
    if any(part in ("", ".", "..") for part in facts["tokenizer_file"][7:].split("/")) or "\\" in facts["tokenizer_file"]:
        raise ArtifactError("setup model facts tokenizer path is invalid")
    for role in ("server", "client"):
        manifest = facts[role + "_manifest"]
        validate("manifest", manifest)
        verify_identity(manifest, facts[role + "_manifest_digest"])
        if manifest["layout_digest"] != facts[role + "_layout_digest"]:
            raise ArtifactError("setup facts layout differs from its manifest")
    server, client = facts["server_manifest"], facts["client_manifest"]
    _identities(server, client)
    for field in ("profile", "logical_model_digest", "operator_contract_digest", "encoding_digest"):
        if facts[field] != server[field]:
            raise ArtifactError("setup facts model identity differs from its manifest")
    if facts["tokenizer_sha256"] != client["tokenizer_digest"]:
        raise ArtifactError("setup facts tokenizer identity differs from its manifest")
    selected = facts["selected_tensors"]
    if not isinstance(selected, list) or any(not isinstance(name, str) for name in selected) or selected != sorted(set(selected)):
        raise ArtifactError("setup facts cache selection is not canonical")
    return facts


def _selected(settings, local_experts=None):
    local = settings.get("client_runtime", {}).get("local_experts", []) if local_experts is None else local_experts
    # This boundary is also called for the peer's plan before a placement
    # exists. Validate exactly the authoritative placement entry schema.
    from .schemas import PLACEMENT_PLAN, StrictValidator
    error = next(StrictValidator(PLACEMENT_PLAN["properties"]["local_experts"]).iter_errors(local), None)
    if error:
        raise ArtifactError("compact local-expert selection: " + error.message)
    keys = [(entry["layer"], entry["expert"]) for entry in local]
    if keys != sorted(set(keys)):
        raise ArtifactError("compact local-expert selection must be sorted and unique")
    return sorted({f"layers.{entry['layer']}.ffn.experts.{entry['expert']}.{projection}.weight"
        for entry in local for projection in ("w1", "w2", "w3")})


def prepare_expert(settings, workspace, *, local_experts=None):
    """Prepare/verify the server and export only its authenticated client subset."""
    selected = _selected(settings, local_experts)
    area = _owned_area(workspace, "models")
    model = settings["model"]
    if model["mode"] == "prepared":
        server_root = _directory(model["package_dir"])
        server = read_json(server_root / "manifest.json", max_bytes=256 << 10)
        verify_identity(server, model["trusted_manifest_digest"])
        server = verify_package(server_root, server)
    elif model["mode"] == "source":
        source = _directory(model["source_dir"])
        provision_source_metadata(source)
        # Besides those fixed metadata originals, preparation writes only its
        # reviewed .ria-recipes subtree of this explicitly writable source.
        private_directory(source / ".ria-recipes")
        recipe = create_recipe(source, settings["profile"], chunk_size=model.get("chunk_size", 4 << 20),
                               max_shard_bytes=model.get("max_shard_bytes", 128 << 30),
                               scratch_bytes=model.get("scratch_bytes", 64 << 20))
        estimate_path = area / (recipe["digest"] + "-estimate.json")
        if estimate_path.exists():
            estimate_document = read_json(estimate_path)
            verify_identity(estimate_document)
            if estimate_document.get("recipe_digest") != recipe["digest"]:
                raise ArtifactError("preparation estimate belongs to another recipe")
        else:
            # Retained Python-object sizes are an observation of this process,
            # rather than a canonical function of the recipe. Keep its first
            # authenticated snapshot on restart instead of fabricating equality.
            estimate_document = seal({"schema_revision": 1, "recipe_digest": recipe["digest"], "estimate": estimate(source, recipe)})
            _publish(estimate_path, estimate_document)
        server_root = _marked_directory(area, "server-" + recipe["digest"], recipe["digest"])
        server = prepare(source, recipe, server_root, role="server")
        runtime_readable(server_root)
    else:
        raise ArtifactError("unknown setup model acquisition/preparation mode")
    if server["role"] != "server" or server["profile"] != settings["profile"]:
        raise ArtifactError("prepared server differs from explicit role/profile")
    key = digest({"server_manifest_digest": server["digest"], "selected_tensors": selected})
    client_root = _marked_directory(area, "client-" + key, key)
    client = client_package(server_root, server["digest"], client_root, selected_names=selected,
                            chunk_size=model.get("chunk_size", 4 << 20),
                            max_shard_bytes=model.get("max_shard_bytes", 128 << 30))
    _identities(server, client)
    tokenizer = _tokenizer(client_root, client)
    runtime_readable(client_root)
    facts = seal({"schema_revision": 1, "kind": "setup_model_facts", "profile": server["profile"],
        "server_root": str(server_root), "client_root": str(client_root), "export_root": str(client_root),
        "server_manifest": server, "client_manifest": client,
        "server_manifest_digest": server["digest"], "client_manifest_digest": client["digest"],
        "logical_model_digest": server["logical_model_digest"], "operator_contract_digest": server["operator_contract_digest"],
        "encoding_digest": server["encoding_digest"], "server_layout_digest": server["layout_digest"],
        "client_layout_digest": client["layout_digest"], "selected_tensors": selected, **tokenizer})
    verify_model_facts(facts)
    return _publish(area / (key + "-facts.json"), facts)


def verify_client_import(root, serverfacts):
    """Bind a transferred client package to its authenticated setup export facts."""
    verify_model_facts(serverfacts)
    root = _directory(root)
    client = read_json(root / "manifest.json", max_bytes=256 << 10)
    verify_identity(client, serverfacts["client_manifest_digest"])
    client = verify_package(root, client)
    _identities(serverfacts["server_manifest"], client)
    provenance = read_json(root / "client-source.json")
    references = [item for item in client["metadata"] if item["path"] == "client-source.json"]
    if len(references) != 1:
        raise ArtifactError("compact client lacks authenticated extraction provenance")
    verify_identity(provenance, references[0]["digest"])
    if provenance.get("server_manifest_digest") != serverfacts["server_manifest_digest"]:
        raise ArtifactError("compact client was extracted from another server root")
    if _tokenizer(root, client) != {field: serverfacts[field] for field in ("tokenizer_file", "tokenizer_sha256", "max_token_bytes")}:
        raise ArtifactError("compact client tokenizer metadata differs from setup facts")
    return client


def make_placement(settings, clientmanifest, serverfacts, workspace):
    """Resolve paths/identities; preserve every explicitly declared runtime pool."""
    verify_model_facts(serverfacts)
    validate("manifest", clientmanifest)
    verify_identity(clientmanifest, serverfacts["client_manifest_digest"])
    _identities(serverfacts["server_manifest"], clientmanifest)
    if _selected(settings) != serverfacts["selected_tensors"]:
        raise ArtifactError("placement local experts differ from prepared compact selection")
    policy = copy.deepcopy(settings["client_runtime"])
    local = policy["local_experts"]
    runtime = {**policy["runtime"], "tokenizer_file": serverfacts["tokenizer_file"],
               "tokenizer_sha256": serverfacts["tokenizer_sha256"], "prefill_rows": settings["planning"]["prefill_rows"]}
    result = seal({"schema_revision": 1, "logical_model_digest": clientmanifest["logical_model_digest"],
        "operator_contract_digest": clientmanifest["operator_contract_digest"], "server_layout_digest": serverfacts["server_layout_digest"],
        "server_executor": settings["server_executor"], "schedule": "full_reference", "shared_placement": "client",
        "expert_policy": "explicit" if local else "remote", "local_experts": local, "runtime": runtime,
        **{name: policy[name] for name in ("host_expert_cache_bytes", "device_expert_cache_bytes", "engram_cache_bytes")}})
    validate("placement-plan", result)
    if settings["profile"] != clientmanifest["profile"] or settings["planning"]["prefill_rows"] > settings["planning"]["context_positions"]:
        raise ArtifactError("placement profile/prefill differs from declared workload")
    _publish(_owned_area(workspace, "config") / "placement-plan.json", result)
    return result


def make_grants(settings, servermanifest, clientfacts, security, workspace):
    """Authorize the finalized client layout/placement and selected transport."""
    validate("manifest", servermanifest)
    verify_identity(servermanifest)
    verify_model_facts(clientfacts)
    if any(clientfacts[field] != servermanifest[field] for field in
           ("profile", "logical_model_digest", "operator_contract_digest", "encoding_digest")) or servermanifest["role"] != "server":
        raise ArtifactError("grant client/server model identities differ")
    if (clientfacts["server_manifest_digest"], clientfacts["server_layout_digest"]) != (servermanifest["digest"], servermanifest["layout_digest"]):
        raise ArtifactError("grant client extraction belongs to another server layout/root")
    peer = security.peer_name("client")
    mode = settings["security"]["mode"]
    if mode not in ("tls", "trusted_network") or security.tls_enabled != (mode == "tls") or (
            mode == "trusted_network" and peer is not None) or (mode == "tls" and (not isinstance(peer, str) or not peer)):
        raise ArtifactError("grant peer identity differs from selected transport")
    result = seal({"schema_revision": 1, "grants": [{"expected_peer_name": peer,
        **{field: servermanifest[field] for field in ("profile", "logical_model_digest", "operator_contract_digest", "encoding_digest")},
        "server_executor": settings["server_executor"], "server_layout_digest": servermanifest["layout_digest"],
        "client_layout_digest": clientfacts["client_layout_digest"], "placement_plan_digest": clientfacts["placement_plan_digest"]}]})
    validate("peer-grants", result)
    return _publish(_owned_area(workspace, "config") / "peer-grants.json", result)


def provision_api_token(workspace, source=None):
    """Return a private token path; never put token bytes or hashes in facts."""
    root = Path(runtime_secret_directory(workspace))
    path = root / "api.token"
    def read_token(candidate, *, owned=False):
        try:
            stream = open_regular(candidate)
        except OSError as exc:
            raise ArtifactError("API token is not an accessible regular file") from exc
        with stream:
            info = os.fstat(stream.fileno())
            if stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1 or not 1 <= info.st_size <= 4096 or (
                    owned and info.st_uid not in (os.geteuid(), RUNTIME_UID)):
                raise ArtifactError("API token must be a private bounded regular file")
            raw = stream.read(4097)
        value = raw.rstrip(b"\r\n")
        if not 1 <= len(value) <= 4096 or any(byte < 33 or byte > 126 for byte in value):
            raise ArtifactError("API token must contain bounded nonwhitespace ASCII")
        return raw
    existing = read_token(path, owned=True) if path.exists() or path.is_symlink() else None
    supplied = read_token(source) if source is not None else None
    if existing is not None:
        if supplied is not None and existing != supplied:
            raise ArtifactError("existing API token differs; explicit workspace rotation required")
    else:
        raw = supplied if supplied is not None else secrets.token_urlsafe(32).encode("ascii") + b"\n"
        with atomic_output(path, mode=0o600, immutable=True) as stream:
            stream.write(raw)
    runtime_secret(path)
    return str(path)
