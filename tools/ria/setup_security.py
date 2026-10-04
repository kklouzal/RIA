"""Invitation-pinned TLS and role-fixed local PKI; trusted TCP creates no PEMs."""

import hashlib
import copy
import ipaddress
import os
from pathlib import Path
import re
import secrets
import ssl
import stat
import subprocess

from .identity import (
    ArtifactError,
    atomic_bytes,
    open_regular,
    loads,
    read_verified_bytes,
)

HEX = re.compile(r"[0-9a-f]{64}\Z")


def endpoint(value):
    if not isinstance(value, str):
        raise ArtifactError("setup endpoint must be a numeric address")
    host, separator, port = value.rpartition(":")
    try:
        address = ipaddress.ip_address(host)
        number = int(port)
    except ValueError as exc:
        raise ArtifactError("setup endpoint must be numeric IP:port") from exc
    if (
        not separator
        or address.version != 4
        or not address.is_private
        or address.is_unspecified
        or address.is_multicast
        or not 1024 <= number <= 65535
        or str(number) != port
    ):
        raise ArtifactError(
            "setup endpoint requires a specific private IPv4 and unprivileged port"
        )
    return str(address), number


def role_name(role, job_id):
    if (
        role not in ("client", "expert")
        or not isinstance(job_id, str)
        or not HEX.fullmatch(job_id)
    ):
        raise ArtifactError("invalid setup role/job identity")
    return "ria-" + role + "." + job_id[:32] + "." + job_id[32:]


def validate_invitation(value):
    fields = {
        "schema_revision",
        "job_id",
        "pair_secret",
        "endpoint",
        "tls_enabled",
        "ca_pem",
        "server_leaf_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or type(value["schema_revision"]) is not int
        or value["schema_revision"] != 1
    ):
        raise ArtifactError("invalid setup invitation fields")
    if (
        not isinstance(value["job_id"], str)
        or not HEX.fullmatch(value["job_id"])
        or not isinstance(value["pair_secret"], str)
        or not HEX.fullmatch(value["pair_secret"])
        or type(value["tls_enabled"]) is not bool
    ):
        raise ArtifactError("invalid setup invitation identity/mode")
    endpoint(value["endpoint"])
    if value["tls_enabled"]:
        if (
            not isinstance(value["ca_pem"], str)
            or len(value["ca_pem"]) > 16384
            or not isinstance(value["server_leaf_sha256"], str)
            or not HEX.fullmatch(value["server_leaf_sha256"])
        ):
            raise ArtifactError(
                "TLS invitation requires its CA and exact server leaf pin"
            )
        try:
            ssl.PEM_cert_to_DER_cert(value["ca_pem"])
        except ValueError as error:
            raise ArtifactError("invalid setup invitation CA") from error
    elif value["ca_pem"] is not None or value["server_leaf_sha256"] is not None:
        raise ArtifactError("trusted-network invitation must not carry certificates")
    return copy.deepcopy(value)


def write_invitation(path, value):
    validate_invitation(value)
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ArtifactError("setup invitation is immutable")
    from .identity import canonical, atomic_output

    with atomic_output(path, mode=0o600, immutable=True) as output:
        output.write(canonical(value) + b"\n")


def read_invitation(path):
    path = Path(path)
    with open_regular(path) as source:
        info = os.fstat(source.fileno())
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ArtifactError("setup invitation must be an owned regular0600 file")
        value = loads(source.read(65537), max_bytes=65536)
        after = os.fstat(source.fileno())
        if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ArtifactError("setup invitation changed during its trusted read")
    return validate_invitation(value)


def _openssl(directory, *arguments):
    result = subprocess.run(
        ["openssl", *arguments],
        cwd=directory,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    if result.returncode or len(result.stdout) > 65536 or len(result.stderr) > 65536:
        raise ArtifactError("bounded setup PKI operation failed")
    return result.stdout


class SetupSecurity:
    def __init__(self, directory, invitation, role):
        self.directory = Path(directory).absolute()
        info = self.directory.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise ArtifactError(
                "setup security requires an owned private real directory"
            )
        self.invitation = validate_invitation(invitation)
        self.role = role
        self.tls_enabled = invitation["tls_enabled"]
        role_name(role, invitation["job_id"])
        self.paths = {
            name: str(self.directory / file)
            for name, file in (
                ("ca_file", "ca.pem"),
                ("certificate_file", "peer.pem"),
                ("private_key_file", "peer.key"),
            )
        }

    @classmethod
    def create_expert(cls, directory, job_id, address, *, tls_enabled=True):
        directory = Path(directory).absolute()
        invitation = {
            "schema_revision": 1,
            "job_id": job_id,
            "pair_secret": secrets.token_hex(32),
            "endpoint": address,
            "tls_enabled": tls_enabled,
            "ca_pem": None,
            "server_leaf_sha256": None,
        }
        role_name("expert", job_id)
        endpoint(address)
        if type(tls_enabled) is not bool:
            raise ArtifactError("setup TLS mode must be explicit Boolean")
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        if tls_enabled:
            _openssl(
                directory,
                "req",
                "-x509",
                "-newkey",
                "ec",
                "-pkeyopt",
                "ec_paramgen_curve:P-256",
                "-nodes",
                "-keyout",
                "ca.key",
                "-out",
                "ca.pem",
                "-days",
                "365",
                "-subj",
                "/CN=RIA setup CA " + job_id[:32],
                "-addext",
                "basicConstraints=critical,CA:TRUE",
                "-addext",
                "keyUsage=critical,keyCertSign,cRLSign",
            )
            _openssl(
                directory,
                "req",
                "-new",
                "-newkey",
                "ec",
                "-pkeyopt",
                "ec_paramgen_curve:P-256",
                "-nodes",
                "-keyout",
                "peer.key",
                "-out",
                "expert.csr",
                "-subj",
                "/CN=RIA expert " + job_id[:32],
            )
            cls._sign(directory, job_id, "expert", "expert.csr", "peer.pem")
            invitation["ca_pem"] = read_verified_bytes(
                directory / "ca.pem", max_bytes=16384
            ).decode("ascii")
            invitation["server_leaf_sha256"] = hashlib.sha256(
                ssl.PEM_cert_to_DER_cert(
                    read_verified_bytes(directory / "peer.pem", max_bytes=16384).decode(
                        "ascii"
                    )
                )
            ).hexdigest()
            for path in directory.iterdir():
                path.chmod(0o600)
        return cls(directory, invitation, "expert"), invitation

    @classmethod
    def create_client(cls, directory, invitation):
        invitation = validate_invitation(invitation)
        directory = Path(directory).absolute()
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        if invitation["tls_enabled"]:
            atomic_bytes(
                directory / "ca.pem", invitation["ca_pem"].encode(), mode=0o600
            )
            _openssl(
                directory,
                "req",
                "-new",
                "-newkey",
                "ec",
                "-pkeyopt",
                "ec_paramgen_curve:P-256",
                "-nodes",
                "-keyout",
                "peer.key",
                "-out",
                "client.csr",
                "-subj",
                "/CN=RIA client " + invitation["job_id"][:32],
            )
            (directory / "peer.key").chmod(0o600)
        return cls(directory, invitation, "client")

    @staticmethod
    def _sign(directory, job_id, role, csr, output):
        extension = (
            "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\n"
            "extendedKeyUsage=serverAuth,clientAuth\nsubjectAltName=DNS:"
            + role_name(role, job_id)
            + "\n"
        )
        atomic_bytes(directory / "role.ext", extension.encode(), mode=0o600)
        _openssl(
            directory,
            "x509",
            "-req",
            "-in",
            csr,
            "-CA",
            "ca.pem",
            "-CAkey",
            "ca.key",
            "-set_serial",
            str(int(secrets.token_hex(16), 16) or 1),
            "-days",
            "365",
            "-out",
            output,
            "-extfile",
            "role.ext",
        )

    def peer_name(self, role):
        return role_name(role, self.invitation["job_id"]) if self.tls_enabled else None

    def client_csr(self):
        if not self.tls_enabled or self.role != "client":
            raise ArtifactError("client CSR requires client TLS policy")
        return read_verified_bytes(
            self.directory / "client.csr", max_bytes=16384
        ).decode("ascii")

    def sign_client_csr(self, csr):
        if (
            not self.tls_enabled
            or self.role != "expert"
            or not isinstance(csr, str)
            or len(csr) > 16384
        ):
            raise ArtifactError("invalid role-fixed client signing operation")
        name = "client-" + hashlib.sha256(csr.encode()).hexdigest()
        csr_path, cert_path = (
            self.directory / (name + ".csr"),
            self.directory / (name + ".pem"),
        )
        if cert_path.exists():
            return read_verified_bytes(cert_path, max_bytes=16384).decode("ascii")
        # A pair provisions one client identity, not a general signing oracle.
        if any(self.directory.glob("client-*.csr")):
            raise ArtifactError(
                "this setup pair already provisioned another client key"
            )
        atomic_bytes(csr_path, csr.encode(), mode=0o600)
        _openssl(self.directory, "req", "-in", csr_path.name, "-noout", "-verify")
        self._sign(
            self.directory,
            self.invitation["job_id"],
            "client",
            csr_path.name,
            cert_path.name,
        )
        cert_path.chmod(0o600)
        return read_verified_bytes(cert_path, max_bytes=16384).decode("ascii")

    def install_client_certificate(self, pem):
        if (
            not self.tls_enabled
            or self.role != "client"
            or not isinstance(pem, str)
            or len(pem) > 16384
        ):
            raise ArtifactError("invalid client certificate installation")
        path = self.directory / "peer.pem"
        if path.exists():
            if path.read_text() != pem:
                raise ArtifactError("client certificate identity is immutable")
            return
        candidate = self.directory / "candidate.pem"
        try:
            atomic_bytes(candidate, pem.encode(), mode=0o600)
            _openssl(
                self.directory,
                "verify",
                "-CAfile",
                "ca.pem",
                "-verify_hostname",
                self.peer_name("client"),
                "candidate.pem",
            )
            # Loading verifies that the returned public leaf matches our local key.
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.load_cert_chain(str(candidate), self.paths["private_key_file"])
            from .identity import atomic_output

            with atomic_output(path, mode=0o600, immutable=True) as output:
                output.write(pem.encode())
        finally:
            candidate.unlink(missing_ok=True)

    def worker_credentials(self):
        """Only the expert leaf/key leave this object; its CA key never does."""
        if not self.tls_enabled:
            return None
        return {
            "certificate": read_verified_bytes(
                self.directory / "peer.pem", max_bytes=16384
            ).decode("ascii"),
            "key": read_verified_bytes(
                self.directory / "peer.key", max_bytes=16384
            ).decode("ascii"),
        }

    def expected_client_leaf_digest(self):
        """The issued client identity is authoritative; never trust a peer claim."""
        if not self.tls_enabled:
            return None
        if self.role != "expert":
            raise ArtifactError("issued client leaf identity belongs to expert policy")
        certificates = list(self.directory.glob("client-*.pem"))
        if len(certificates) != 1:
            raise ArtifactError(
                "setup requires exactly one actually issued client leaf"
            )
        pem = read_verified_bytes(certificates[0], max_bytes=16384).decode("ascii")
        return hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest()
