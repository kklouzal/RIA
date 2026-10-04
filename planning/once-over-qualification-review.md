# Qualification and requirement coverage once-over

Baseline: `fc7c5d86e59b26376fd0191ede120d34526def34`. Reviewed the original
specification's oracle, fidelity, matrix, reproducibility and release requirements
(18–22), implementation-plan gate mapping, current producer/readiness catalog,
saved-logit comparison and calibration admission. The original specification
SHA256 remains `14224cdb33476944111e14f69a5679f0597c192a44d67b48f048910f326f3f6e`.

The saved-logit comparator hashed pathnames before and after opening separate
comparison readers. An ABA replacement could make both pathname hashes agree
while the readers consumed different logits. A deterministic synthetic
reproduction passed an exact comparison while recording the checksum of the
unread, changed candidate. The original evidence is retained in
`build/ria/evidence/onceover-logits-aba-reproduction.log`.

The correction hashes the exact header and rows consumed from held regular-file
descriptors. Descriptor/path fingerprints cover inode, size, mtime and ctime
before/after comparison and again after closing readers. Dimension/size checks
reject incomplete or appended artifacts. Evidence hashes now identify the same
bytes as the metrics. Numerical thresholds, shifted labels, FP64 stable NLL,
operator/model identities and bounded vocabulary-row scratch are preserved.
This removes redundant input passes; target throughput improvements are unmeasured.

Non-object teacher input and policy JSON previously raised uncaught `TypeError`
before schema validation. They now produce the tool's typed boundary error and
publish no output. Forty focused comparison/calibration tests passed, including
path restoration, in-place restoration with restored mtime, identical-byte inode
replacement, independently recomputed file hashes, malformed top-level values,
all existing numerical/identity failures and release-admission refusals. Full
log: `build/ria/evidence/onceover-qualification-targeted.log`.

The broader qualification software gap remains open. Existing initial fixture
registration derives five components from bounded native/transport reports and
supervision. HTTP replay measures its explicit workloads and soak. Saved-logit
comparison measures its supplied corpus. These scopes do not implement the
full target checkpoint/reference producer, all 540 semantic matrix cells,
gate-specific observations, fault injection or independent full-model state
oracles. Generic numeric diagnostics grant no gate or cell credit; final
release admission fails closed.

The acceptance policy freezes source/model identity, thresholds, ordered
objectives and soak minimum. It does not supply a full-model registration that
binds the reviewed oracle, corpus, selected positions, profile and determinism
contract before candidate results. The readiness catalog now names that missing
producer explicitly alongside the missing execution producer. Reviewed policy
and corpus inputs also remain required; those external inputs alone would not
implement the missing software.

`tools/qualify_ria.py readiness` regenerated
`planning/qualification-software-readiness.json`, returning the expected exit 1
with `release_ready=false`. This is an explicit failed readiness gate, not a
passed test or a claim that hardware alone completes the project. Full software
implementation and final release qualification remain false. Physical GPU,
full-model/source fidelity, target memory/NUMA, performance and soak execution
remain excluded by the user's no-live-testing instruction.
