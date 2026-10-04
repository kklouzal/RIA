# Delivery and artifact once-over

Baseline: `fc7c5d8` (`97310cd` implements prompt prefill). Scope: specification
sections 4–6, 15, 19 and 22; artifact/import identity, preparation/extraction,
native metadata admission, deployment, schemas, container and CI contracts.
No checkpoint acquisition, GPU query/initialization/kernel, target probe,
model inference, physical deployment, benchmark or soak was performed.

## Findings corrected

1. **Output-path escape before fail-closed validation (integrity/security).**
   A valid synthetic preparation with `output/index` linked to a sibling
   directory wrote the sibling's `0.json`, then failed final package validation.
   Preparation and compact-client extraction now inspect every owned output
   directory before mutation. Shared publication traverses directories with
   `O_DIRECTORY|O_NOFOLLOW` and held directory descriptors, creates entries with
   parent `fsync`, and stages/replaces files with descriptor-relative syscalls.
   Target recipe and Engram metadata directory creation use that same boundary.
   Immutable shard publication uses a non-replacing hard link; existing bytes
   must have the temporary's exact size, compare in bounded blocks, and retain
   the same descriptor/path identities and timestamps through comparison.
   Cleanup never deletes a colliding unowned temporary and preserves the
   primary exception with additional cleanup context.

2. **Official converter root/operator revision mismatch (handoff blocker).**
   The automatic converter assigned NVIDIA revision `3431dde...` to the
   recipe/root, while its operator contract identified native revision
   `2cba9e42...`; the strict native loader rejects that combination. One owned
   identity constructor now binds the recipe/root/operator to the native graph
   revision, while preserving the independent publisher revision in logical
   identity and calibration provenance. Preparation checks source-revision
   consistency before creating output. An actual native-loader fixture accepts
   the new constructed identity and continues to reject the former mismatch.

3. **Metadata transient allocation underestimate (resource correctness).**
   `tensor.c` duplicated the old parser allocation equation and omitted
   duplicate-key arrays and canonical/hash output/key arrays. The loader now
   calls the same pure equations used by the production JSON implementation:
   input/DOM/duplicate-key peak before parsing, then actual retained-DOM and
   canonical/key peak before identity hashing. Borrowed safetensors header bytes
   do not become a second copied-input allocation. A 256 KiB large-string
   document requires a larger canonical peak than its input/DOM peak and is
   rejected with an empty cleaned DOM under the smaller reservation.

4. **FIFO stalls before regular-file validation (availability).**
   The native manifest and artifact leaf opens lacked `O_NONBLOCK`; a FIFO
   could block before `fstat` rejected its type. They now open nonblocking and
   retain all regular-file/size checks. Isolated manifest, index and shard FIFO
   fixtures return typed failure within a two-second test deadline.

## Focused evidence

- Strict scratch native loader build, no shared Makefile/object mutation.
- `test_artifacts.py`: **46 passed**, including directory links, ancestor links,
  path replacement after parent opening, immutable inode/mutate-restore,
  collision ownership, durability and primary-error cleanup contracts.
- `test_native_artifacts.py`: **20 passed**, using the scratch production loader,
  including source identities and all three FIFO boundaries.
- Actual `test_tensor_metadata.c`: strict normal and ASan/UBSan/leak executions
  passed at 128-byte, 16 KiB and 256 KiB inputs, precise allocation boundaries,
  canonical hashing and rejected-allocation cleanup. Root integrated this fixture
  into canonical `ria-offline` and `ria-sanitize`; the superseded Python scratch
  wrapper was removed.
- Deployment/inventory/host/container-inspection/build-contract suites:
  **85 passed** against existing native inventory executable (not a claim that
  the changed native inventory integration was rebuilt in this delegated run).
- Owned Python files and regressions: Ruff passed; owned diff whitespace passed.

## Coverage and limits

- Profiles and preparation: reviewed full-population calibration reductions,
  source-index/header/LFS agreement, noncircular native/publisher identity,
  bounded conversion, chunk-boundary grants, compact-role closure, source
  mutation snapshots, manifest-last publication and restart checkpoints.
- Admission: reviewed the shared native inventory path, NUMA reservation/replica
  accounting, every phase, required `prefill_rows`, projection/remote/pinned
  floors, provenance rederivation, role restrictions and immutable rollout.
- Deployment/security: reviewed strict duplicate/unknown-key schemas, controlled
  subprocess environment and deadlines, role controller lock, bootstrap/finalize,
  sealed raw evidence, effective Compose/inspection equality, host ancestry,
  exact selected UUID, read-only mounts, private credentials, finite no-swap and
  memlock, bounded tmpfs, non-root entrypoints and seccomp generation.
- Delivery: reviewed hash-pinned source/base/tool inputs, numerical flags,
  SM120a AOT/no-PTX proof path, CPU independence, build-info/source identities,
  model/secret exclusion, SBOM/provenance and authorized GHCR workflow behavior.
- This review does **not** close the separately documented complete physical
  qualification producer/matrix gap. Built images and metadata/static fixtures
  do not prove physical readiness, mathematical fidelity, NUMA placement,
  container runtime permissions, performance or soak. Root owns that review and
  the final frozen build/test/static/sanitizer/hosted publication evidence.

Owned source changes are ready for the root's frozen integration gates.
