"""
mTLS (Mutual TLS) Module for Claude Multi-Agent Coordination System.

Provides mutual TLS support for secure inter-agent communication.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import json
import hashlib

# Try to import cryptography library for certificate operations
try:
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, ec
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


@dataclass
class CertificateInfo:
    """Information about a certificate."""
    common_name: str
    subject: str
    issuer: str
    serial_number: str
    not_before: str
    not_after: str
    fingerprint: str
    is_ca: bool = False
    agent_id: Optional[str] = None
    role: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'common_name': self.common_name,
            'subject': self.subject,
            'issuer': self.issuer,
            'serial_number': self.serial_number,
            'not_before': self.not_before,
            'not_after': self.not_after,
            'fingerprint': self.fingerprint,
            'is_ca': self.is_ca,
            'agent_id': self.agent_id,
            'role': self.role,
        }

    def is_valid(self) -> bool:
        """Check if certificate is currently valid."""
        now = datetime.now()
        not_before = datetime.fromisoformat(self.not_before.replace('Z', '+00:00').replace('+00:00', ''))
        not_after = datetime.fromisoformat(self.not_after.replace('Z', '+00:00').replace('+00:00', ''))
        return not_before <= now <= not_after


@dataclass
class MTLSConfig:
    """Configuration for mTLS."""
    ca_cert_path: str
    ca_key_path: str
    cert_path: str
    key_path: str
    verify_client: bool = True
    verify_server: bool = True
    cert_validity_days: int = 365
    key_type: str = "rsa"  # "rsa" or "ec"
    key_size: int = 4096  # For RSA
    ec_curve: str = "secp384r1"  # For EC


class MTLSManager:
    """
    Manages mTLS certificates for inter-agent communication.

    Features:
    - CA certificate generation
    - Agent certificate generation
    - Certificate validation
    - Certificate rotation
    - Certificate revocation
    """

    def __init__(
        self,
        cert_dir: str,
        ca_common_name: str = "Claude Multi-Agent CA",
        config: Optional[MTLSConfig] = None,
    ):
        """
        Initialize the mTLS manager.

        Args:
            cert_dir: Directory to store certificates
            ca_common_name: Common name for the CA certificate
            config: Optional configuration override
        """
        self.cert_dir = Path(cert_dir)
        self.ca_common_name = ca_common_name
        self.cert_dir.mkdir(parents=True, exist_ok=True)

        # Default config
        self.config = config or MTLSConfig(
            ca_cert_path=str(self.cert_dir / "ca.crt"),
            ca_key_path=str(self.cert_dir / "ca.key"),
            cert_path=str(self.cert_dir / "server.crt"),
            key_path=str(self.cert_dir / "server.key"),
        )

        # Track issued certificates
        self._issued_certs: Dict[str, CertificateInfo] = {}
        self._revoked_serials: set = set()
        self._certs_db_path = self.cert_dir / "certs.json"

        self._load_certs_db()

    def _load_certs_db(self) -> None:
        """Load certificates database."""
        if self._certs_db_path.exists():
            try:
                with open(self._certs_db_path, 'r') as f:
                    data = json.load(f)
                for serial, cert_data in data.get('certificates', {}).items():
                    self._issued_certs[serial] = CertificateInfo(**cert_data)
                self._revoked_serials = set(data.get('revoked', []))
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_certs_db(self) -> None:
        """Save certificates database."""
        data = {
            'certificates': {
                serial: cert.to_dict()
                for serial, cert in self._issued_certs.items()
            },
            'revoked': list(self._revoked_serials),
        }
        with open(self._certs_db_path, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_ca(self, validity_days: int = 3650) -> Tuple[str, str]:
        """
        Generate a CA certificate.

        Returns: (ca_cert_path, ca_key_path)
        """
        if CRYPTO_AVAILABLE:
            return self._generate_ca_cryptography(validity_days)
        else:
            return self._generate_ca_openssl(validity_days)

    def _generate_ca_cryptography(self, validity_days: int) -> Tuple[str, str]:
        """Generate CA using cryptography library."""
        # Generate private key
        if self.config.key_type == "ec":
            private_key = ec.generate_private_key(
                ec.SECP384R1(),
                default_backend()
            )
        else:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.config.key_size,
                backend=default_backend()
            )

        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, self.ca_common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Claude Multi-Agent System"),
        ])

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=validity_days))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(private_key, hashes.SHA256(), default_backend())
        )

        # Save certificate
        cert_path = self.config.ca_cert_path
        key_path = self.config.ca_key_path

        with open(cert_path, 'wb') as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        with open(key_path, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # Track CA cert
        cert_info = CertificateInfo(
            common_name=self.ca_common_name,
            subject=self.ca_common_name,
            issuer=self.ca_common_name,
            serial_number=str(cert.serial_number),
            not_before=cert.not_valid_before_utc.isoformat(),
            not_after=cert.not_valid_after_utc.isoformat(),
            fingerprint=cert.fingerprint(hashes.SHA256()).hex(),
            is_ca=True,
        )
        self._issued_certs[cert_info.serial_number] = cert_info
        self._save_certs_db()

        return cert_path, key_path

    def _generate_ca_openssl(self, validity_days: int) -> Tuple[str, str]:
        """Generate CA using OpenSSL command line."""
        cert_path = self.config.ca_cert_path
        key_path = self.config.ca_key_path

        # Generate private key
        subprocess.run([
            'openssl', 'genrsa', '-out', key_path, str(self.config.key_size)
        ], check=True, capture_output=True)

        # Generate certificate
        subprocess.run([
            'openssl', 'req', '-new', '-x509',
            '-key', key_path,
            '-out', cert_path,
            '-days', str(validity_days),
            '-subj', f'/CN={self.ca_common_name}/O=Claude Multi-Agent System'
        ], check=True, capture_output=True)

        # Parse cert info
        cert_info = self._parse_cert_openssl(cert_path)
        cert_info.is_ca = True
        self._issued_certs[cert_info.serial_number] = cert_info
        self._save_certs_db()

        return cert_path, key_path

    def generate_agent_cert(
        self,
        agent_id: str,
        role: str = "worker",
        validity_days: Optional[int] = None,
        sans: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """
        Generate a certificate for an agent.

        Args:
            agent_id: The agent's identifier
            role: The agent's role (leader, worker, admin)
            validity_days: Certificate validity in days
            sans: Subject Alternative Names (DNS names, IPs)

        Returns: (cert_path, key_path)
        """
        validity = validity_days or self.config.cert_validity_days

        if CRYPTO_AVAILABLE:
            return self._generate_agent_cert_cryptography(agent_id, role, validity, sans)
        else:
            return self._generate_agent_cert_openssl(agent_id, role, validity, sans)

    def _generate_agent_cert_cryptography(
        self,
        agent_id: str,
        role: str,
        validity_days: int,
        sans: Optional[List[str]],
    ) -> Tuple[str, str]:
        """Generate agent certificate using cryptography library."""
        # Load CA
        with open(self.config.ca_cert_path, 'rb') as f:
            ca_cert = x509.load_pem_x509_certificate(f.read(), default_backend())
        with open(self.config.ca_key_path, 'rb') as f:
            ca_key = serialization.load_pem_private_key(f.read(), None, default_backend())

        # Generate agent key
        if self.config.key_type == "ec":
            private_key = ec.generate_private_key(
                ec.SECP384R1(),
                default_backend()
            )
        else:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.config.key_size,
                backend=default_backend()
            )

        # Build certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, agent_id),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Claude Multi-Agent System"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, role),
        ])

        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=validity_days))
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([
                    x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                ]),
                critical=False,
            )
        )

        # Add SANs
        if sans:
            san_list = []
            for san in sans:
                if san.replace('.', '').isdigit():  # IP address
                    from ipaddress import ip_address
                    san_list.append(x509.IPAddress(ip_address(san)))
                else:
                    san_list.append(x509.DNSName(san))
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False,
            )

        cert = builder.sign(ca_key, hashes.SHA256(), default_backend())

        # Save certificate and key
        cert_path = str(self.cert_dir / f"{agent_id}.crt")
        key_path = str(self.cert_dir / f"{agent_id}.key")

        with open(cert_path, 'wb') as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        with open(key_path, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # Track certificate
        cert_info = CertificateInfo(
            common_name=agent_id,
            subject=agent_id,
            issuer=self.ca_common_name,
            serial_number=str(cert.serial_number),
            not_before=cert.not_valid_before_utc.isoformat(),
            not_after=cert.not_valid_after_utc.isoformat(),
            fingerprint=cert.fingerprint(hashes.SHA256()).hex(),
            is_ca=False,
            agent_id=agent_id,
            role=role,
        )
        self._issued_certs[cert_info.serial_number] = cert_info
        self._save_certs_db()

        return cert_path, key_path

    def _generate_agent_cert_openssl(
        self,
        agent_id: str,
        role: str,
        validity_days: int,
        sans: Optional[List[str]],
    ) -> Tuple[str, str]:
        """Generate agent certificate using OpenSSL command line."""
        cert_path = str(self.cert_dir / f"{agent_id}.crt")
        key_path = str(self.cert_dir / f"{agent_id}.key")
        csr_path = str(self.cert_dir / f"{agent_id}.csr")

        # Generate private key
        subprocess.run([
            'openssl', 'genrsa', '-out', key_path, str(self.config.key_size)
        ], check=True, capture_output=True)

        # Generate CSR
        subject = f'/CN={agent_id}/O=Claude Multi-Agent System/OU={role}'
        subprocess.run([
            'openssl', 'req', '-new',
            '-key', key_path,
            '-out', csr_path,
            '-subj', subject
        ], check=True, capture_output=True)

        # Sign with CA
        cmd = [
            'openssl', 'x509', '-req',
            '-in', csr_path,
            '-CA', self.config.ca_cert_path,
            '-CAkey', self.config.ca_key_path,
            '-CAcreateserial',
            '-out', cert_path,
            '-days', str(validity_days),
        ]

        # Add SANs if provided
        if sans:
            san_config = f"subjectAltName = {','.join(f'DNS:{s}' if not s.replace('.','').isdigit() else f'IP:{s}' for s in sans)}"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
                f.write(san_config)
                san_file = f.name
            cmd.extend(['-extfile', san_file])

        subprocess.run(cmd, check=True, capture_output=True)

        # Clean up CSR
        os.remove(csr_path)
        if sans:
            os.remove(san_file)

        # Parse and track cert
        cert_info = self._parse_cert_openssl(cert_path)
        cert_info.agent_id = agent_id
        cert_info.role = role
        self._issued_certs[cert_info.serial_number] = cert_info
        self._save_certs_db()

        return cert_path, key_path

    def _parse_cert_openssl(self, cert_path: str) -> CertificateInfo:
        """Parse certificate info using OpenSSL."""
        result = subprocess.run([
            'openssl', 'x509', '-in', cert_path,
            '-noout', '-subject', '-issuer', '-serial',
            '-startdate', '-enddate', '-fingerprint'
        ], capture_output=True, text=True, check=True)

        lines = result.stdout.strip().split('\n')
        info = {}
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                info[key.strip().lower()] = value.strip()

        return CertificateInfo(
            common_name=info.get('cn', info.get('subject', '')),
            subject=info.get('subject', ''),
            issuer=info.get('issuer', ''),
            serial_number=info.get('serial', ''),
            not_before=info.get('notbefore', ''),
            not_after=info.get('notafter', ''),
            fingerprint=info.get('sha256 fingerprint', info.get('fingerprint', '')),
        )

    def verify_certificate(
        self,
        cert_path: str,
        check_revocation: bool = True,
    ) -> Tuple[bool, Optional[CertificateInfo], str]:
        """
        Verify a certificate.

        Returns: (is_valid, cert_info, error_message)
        """
        try:
            if CRYPTO_AVAILABLE:
                return self._verify_cert_cryptography(cert_path, check_revocation)
            else:
                return self._verify_cert_openssl(cert_path, check_revocation)
        except Exception as e:
            return False, None, str(e)

    def _verify_cert_cryptography(
        self,
        cert_path: str,
        check_revocation: bool,
    ) -> Tuple[bool, Optional[CertificateInfo], str]:
        """Verify certificate using cryptography library."""
        with open(cert_path, 'rb') as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())

        # Check validity period
        now = datetime.utcnow()
        if cert.not_valid_before_utc > now:
            return False, None, "Certificate not yet valid"
        if cert.not_valid_after_utc < now:
            return False, None, "Certificate has expired"

        # Check if revoked
        serial = str(cert.serial_number)
        if check_revocation and serial in self._revoked_serials:
            return False, None, "Certificate has been revoked"

        # Verify signature against CA
        with open(self.config.ca_cert_path, 'rb') as f:
            ca_cert = x509.load_pem_x509_certificate(f.read(), default_backend())

        try:
            ca_cert.public_key().verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                cert.signature_algorithm_parameters,
            )
        except Exception:
            return False, None, "Certificate signature verification failed"

        # Build cert info
        cn = ""
        ou = ""
        for attr in cert.subject:
            if attr.oid == NameOID.COMMON_NAME:
                cn = attr.value
            elif attr.oid == NameOID.ORGANIZATIONAL_UNIT_NAME:
                ou = attr.value

        cert_info = CertificateInfo(
            common_name=cn,
            subject=cert.subject.rfc4514_string(),
            issuer=cert.issuer.rfc4514_string(),
            serial_number=serial,
            not_before=cert.not_valid_before_utc.isoformat(),
            not_after=cert.not_valid_after_utc.isoformat(),
            fingerprint=cert.fingerprint(hashes.SHA256()).hex(),
            is_ca=False,
            agent_id=cn,
            role=ou,
        )

        return True, cert_info, ""

    def _verify_cert_openssl(
        self,
        cert_path: str,
        check_revocation: bool,
    ) -> Tuple[bool, Optional[CertificateInfo], str]:
        """Verify certificate using OpenSSL."""
        # Verify against CA
        result = subprocess.run([
            'openssl', 'verify',
            '-CAfile', self.config.ca_cert_path,
            cert_path
        ], capture_output=True, text=True)

        if result.returncode != 0:
            return False, None, result.stderr

        cert_info = self._parse_cert_openssl(cert_path)

        # Check revocation
        if check_revocation and cert_info.serial_number in self._revoked_serials:
            return False, cert_info, "Certificate has been revoked"

        return True, cert_info, ""

    def revoke_certificate(self, serial_number: str) -> bool:
        """Revoke a certificate by its serial number."""
        if serial_number in self._issued_certs:
            self._revoked_serials.add(serial_number)
            self._save_certs_db()
            return True
        return False

    def revoke_agent_cert(self, agent_id: str) -> bool:
        """Revoke all certificates for an agent."""
        revoked = False
        for serial, cert_info in self._issued_certs.items():
            if cert_info.agent_id == agent_id:
                self._revoked_serials.add(serial)
                revoked = True
        if revoked:
            self._save_certs_db()
        return revoked

    def list_certificates(self, include_revoked: bool = False) -> List[CertificateInfo]:
        """List all issued certificates."""
        certs = []
        for serial, cert_info in self._issued_certs.items():
            if not include_revoked and serial in self._revoked_serials:
                continue
            certs.append(cert_info)
        return certs

    def get_certificate_info(self, serial_number: str) -> Optional[CertificateInfo]:
        """Get certificate info by serial number."""
        return self._issued_certs.get(serial_number)

    def is_certificate_revoked(self, serial_number: str) -> bool:
        """Check if a certificate is revoked."""
        return serial_number in self._revoked_serials

    def rotate_agent_cert(
        self,
        agent_id: str,
        role: str = "worker",
        validity_days: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Rotate an agent's certificate.

        Revokes the old certificate and generates a new one.
        """
        # Revoke old cert
        self.revoke_agent_cert(agent_id)

        # Generate new cert
        return self.generate_agent_cert(agent_id, role, validity_days)

    def get_ssl_context(self, is_server: bool = False) -> Any:
        """
        Get an SSL context configured for mTLS.

        Returns ssl.SSLContext if ssl module is available.
        """
        import ssl

        if is_server:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.verify_mode = ssl.CERT_REQUIRED if self.config.verify_client else ssl.CERT_OPTIONAL
        else:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = False  # We do our own CN verification
            context.verify_mode = ssl.CERT_REQUIRED if self.config.verify_server else ssl.CERT_NONE

        # Load CA
        context.load_verify_locations(self.config.ca_cert_path)

        # Load certificate and key
        context.load_cert_chain(
            certfile=self.config.cert_path,
            keyfile=self.config.key_path,
        )

        return context
