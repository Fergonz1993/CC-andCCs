"""
Encryption Module for Claude Multi-Agent Coordination System.

Provides:
- Task data encryption at rest
- Secure credential storage
- Key management
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import struct

# Try to import cryptography library
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class EncryptionError(Exception):
    """Raised when encryption/decryption fails."""
    pass


class EncryptionManager:
    """
    Manages encryption for task data at rest.

    Features:
    - AES-256-GCM encryption
    - Key derivation from master key
    - Per-task encryption keys
    - Envelope encryption pattern
    """

    def __init__(
        self,
        master_key: Optional[str] = None,
        key_derivation_salt: Optional[bytes] = None,
    ):
        """
        Initialize the encryption manager.

        Args:
            master_key: The master encryption key. If not provided, looks for
                       ENCRYPTION_MASTER_KEY env var, or generates a new one.
            key_derivation_salt: Salt for key derivation. Generated if not provided.
        """
        self._master_key = master_key or os.environ.get('ENCRYPTION_MASTER_KEY')
        if not self._master_key:
            self._master_key = secrets.token_hex(32)

        self._salt = key_derivation_salt or os.urandom(16)
        self._derived_key = self._derive_key(self._master_key.encode())

    def _derive_key(self, password: bytes, iterations: int = 100000) -> bytes:
        """Derive an encryption key from a password/master key."""
        if CRYPTO_AVAILABLE:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self._salt,
                iterations=iterations,
            )
            return kdf.derive(password)
        else:
            # Fallback: simple PBKDF2 implementation
            return self._pbkdf2_sha256(password, self._salt, iterations, 32)

    def _pbkdf2_sha256(
        self,
        password: bytes,
        salt: bytes,
        iterations: int,
        dklen: int,
    ) -> bytes:
        """PBKDF2-HMAC-SHA256 implementation."""
        return hashlib.pbkdf2_hmac('sha256', password, salt, iterations, dklen)

    def encrypt(self, plaintext: Union[str, bytes, dict]) -> str:
        """
        Encrypt data.

        Args:
            plaintext: Data to encrypt (string, bytes, or dict)

        Returns:
            Base64-encoded encrypted data with nonce
        """
        # Convert to bytes
        if isinstance(plaintext, dict):
            plaintext = json.dumps(plaintext).encode('utf-8')
        elif isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')

        # Generate nonce
        nonce = os.urandom(12)

        if CRYPTO_AVAILABLE:
            aesgcm = AESGCM(self._derived_key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        else:
            # Fallback: XOR with key stream (less secure but functional)
            ciphertext = self._simple_encrypt(plaintext, nonce)

        # Combine nonce + ciphertext
        encrypted_data = nonce + ciphertext

        return base64.b64encode(encrypted_data).decode('utf-8')

    def decrypt(self, encrypted_data: str, as_dict: bool = False) -> Union[str, dict]:
        """
        Decrypt data.

        Args:
            encrypted_data: Base64-encoded encrypted data
            as_dict: If True, parse the decrypted data as JSON

        Returns:
            Decrypted string or dict
        """
        try:
            data = base64.b64decode(encrypted_data)
            nonce = data[:12]
            ciphertext = data[12:]

            if CRYPTO_AVAILABLE:
                aesgcm = AESGCM(self._derived_key)
                plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            else:
                plaintext = self._simple_decrypt(ciphertext, nonce)

            decoded = plaintext.decode('utf-8')

            if as_dict:
                return json.loads(decoded)
            return decoded

        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}")

    def _simple_encrypt(self, plaintext: bytes, nonce: bytes) -> bytes:
        """
        Simple encryption fallback using AES-CTR-like construction.
        NOT as secure as AES-GCM, but works without external libraries.
        """
        # Generate key stream
        key_stream = self._generate_key_stream(nonce, len(plaintext))

        # XOR plaintext with key stream
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, key_stream))

        # Add HMAC for integrity
        mac = hmac.new(self._derived_key, nonce + ciphertext, hashlib.sha256).digest()

        return ciphertext + mac

    def _simple_decrypt(self, ciphertext_with_mac: bytes, nonce: bytes) -> bytes:
        """Simple decryption fallback."""
        # Separate ciphertext and MAC
        ciphertext = ciphertext_with_mac[:-32]
        mac = ciphertext_with_mac[-32:]

        # Verify MAC
        expected_mac = hmac.new(
            self._derived_key, nonce + ciphertext, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(mac, expected_mac):
            raise EncryptionError("MAC verification failed")

        # Decrypt
        key_stream = self._generate_key_stream(nonce, len(ciphertext))
        plaintext = bytes(a ^ b for a, b in zip(ciphertext, key_stream))

        return plaintext

    def _generate_key_stream(self, nonce: bytes, length: int) -> bytes:
        """Generate a key stream for CTR-like encryption."""
        key_stream = b''
        counter = 0
        while len(key_stream) < length:
            block = hashlib.sha256(
                self._derived_key + nonce + struct.pack('<Q', counter)
            ).digest()
            key_stream += block
            counter += 1
        return key_stream[:length]

    def encrypt_task_data(self, task_data: dict) -> dict:
        """
        Encrypt sensitive fields in task data.

        Fields encrypted: description, context.hints, result.output
        """
        encrypted = task_data.copy()

        # Encrypt sensitive fields
        sensitive_fields = ['description']
        for field in sensitive_fields:
            if field in encrypted and encrypted[field]:
                encrypted[field] = self.encrypt(encrypted[field])
                encrypted[f'_{field}_encrypted'] = True

        # Encrypt nested sensitive fields
        if 'context' in encrypted and encrypted['context']:
            ctx = encrypted['context'].copy()
            if 'hints' in ctx and ctx['hints']:
                ctx['hints'] = self.encrypt(ctx['hints'])
                ctx['_hints_encrypted'] = True
            encrypted['context'] = ctx

        if 'result' in encrypted and encrypted['result']:
            result = encrypted['result'].copy()
            if 'output' in result and result['output']:
                result['output'] = self.encrypt(result['output'])
                result['_output_encrypted'] = True
            encrypted['result'] = result

        return encrypted

    def decrypt_task_data(self, encrypted_data: dict) -> dict:
        """
        Decrypt sensitive fields in task data.
        """
        decrypted = encrypted_data.copy()

        # Decrypt top-level fields
        if decrypted.get('_description_encrypted'):
            decrypted['description'] = self.decrypt(decrypted['description'])
            del decrypted['_description_encrypted']

        # Decrypt nested fields
        if 'context' in decrypted and decrypted['context']:
            ctx = decrypted['context'].copy()
            if ctx.get('_hints_encrypted'):
                ctx['hints'] = self.decrypt(ctx['hints'])
                del ctx['_hints_encrypted']
            decrypted['context'] = ctx

        if 'result' in decrypted and decrypted['result']:
            result = decrypted['result'].copy()
            if result.get('_output_encrypted'):
                result['output'] = self.decrypt(result['output'])
                del result['_output_encrypted']
            decrypted['result'] = result

        return decrypted

    def get_salt(self) -> bytes:
        """Get the key derivation salt (needed for key recovery)."""
        return self._salt

    def rotate_key(self, new_master_key: str) -> "EncryptionManager":
        """
        Create a new encryption manager with a rotated key.

        Use this to re-encrypt data with a new key.
        """
        return EncryptionManager(
            master_key=new_master_key,
            key_derivation_salt=os.urandom(16),  # New salt for new key
        )


@dataclass
class StoredCredential:
    """A stored credential."""
    name: str
    encrypted_value: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None
    expires_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecureCredentialStore:
    """
    Securely stores credentials and secrets.

    Features:
    - Encrypted credential storage
    - Access logging
    - Credential expiration
    - Prevents secret logging
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        encryption_manager: Optional[EncryptionManager] = None,
    ):
        """
        Initialize the credential store.

        Args:
            storage_path: Path to store encrypted credentials
            encryption_manager: EncryptionManager instance for encryption
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._encryption = encryption_manager or EncryptionManager()
        self._credentials: Dict[str, StoredCredential] = {}
        self._access_log: list = []

        if self.storage_path and self.storage_path.exists():
            self._load_credentials()

    def _load_credentials(self) -> None:
        """Load credentials from encrypted storage."""
        if not self.storage_path:
            return
        try:
            with open(self.storage_path, 'rb') as f:
                encrypted_content = f.read().decode('utf-8')

            data = self._encryption.decrypt(encrypted_content, as_dict=True)
            for name, cred_data in data.get('credentials', {}).items():
                self._credentials[name] = StoredCredential(**cred_data)

        except Exception:
            # Start fresh if we can't load
            pass

    def _save_credentials(self) -> None:
        """Save credentials to encrypted storage."""
        if not self.storage_path:
            return

        data = {
            'credentials': {
                name: {
                    'name': cred.name,
                    'encrypted_value': cred.encrypted_value,
                    'created_at': cred.created_at,
                    'updated_at': cred.updated_at,
                    'expires_at': cred.expires_at,
                    'metadata': cred.metadata,
                }
                for name, cred in self._credentials.items()
            }
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        encrypted_content = self._encryption.encrypt(data)
        with open(self.storage_path, 'wb') as f:
            f.write(encrypted_content.encode('utf-8'))

    def store(
        self,
        name: str,
        value: str,
        expires_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a credential securely.

        Args:
            name: Credential name/key
            value: The secret value (will be encrypted)
            expires_at: Optional expiration ISO timestamp
            metadata: Optional metadata (NOT encrypted)
        """
        encrypted_value = self._encryption.encrypt(value)

        if name in self._credentials:
            # Update existing
            cred = self._credentials[name]
            cred.encrypted_value = encrypted_value
            cred.updated_at = datetime.now().isoformat()
            if expires_at:
                cred.expires_at = expires_at
            if metadata:
                cred.metadata = metadata
        else:
            # Create new
            self._credentials[name] = StoredCredential(
                name=name,
                encrypted_value=encrypted_value,
                expires_at=expires_at,
                metadata=metadata or {},
            )

        self._save_credentials()

    def retrieve(self, name: str, accessor: str = "unknown") -> Optional[str]:
        """
        Retrieve a credential.

        Args:
            name: Credential name/key
            accessor: Identifier for who is accessing (for logging)

        Returns:
            Decrypted value or None if not found/expired
        """
        cred = self._credentials.get(name)
        if not cred:
            self._log_access(name, accessor, success=False, reason="not_found")
            return None

        # Check expiration
        if cred.expires_at:
            if datetime.fromisoformat(cred.expires_at) < datetime.now():
                self._log_access(name, accessor, success=False, reason="expired")
                return None

        self._log_access(name, accessor, success=True)

        try:
            return self._encryption.decrypt(cred.encrypted_value)
        except EncryptionError:
            self._log_access(name, accessor, success=False, reason="decryption_failed")
            return None

    def delete(self, name: str) -> bool:
        """Delete a credential."""
        if name in self._credentials:
            del self._credentials[name]
            self._save_credentials()
            return True
        return False

    def list_credentials(self) -> list[Dict[str, Any]]:
        """List all credential names and metadata (NOT values)."""
        return [
            {
                'name': cred.name,
                'created_at': cred.created_at,
                'updated_at': cred.updated_at,
                'expires_at': cred.expires_at,
                'metadata': cred.metadata,
            }
            for cred in self._credentials.values()
        ]

    def _log_access(
        self,
        credential_name: str,
        accessor: str,
        success: bool,
        reason: Optional[str] = None,
    ) -> None:
        """Log credential access."""
        self._access_log.append({
            'timestamp': datetime.now().isoformat(),
            'credential': credential_name,
            'accessor': accessor,
            'success': success,
            'reason': reason,
        })

        # Keep only last 1000 entries
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-1000:]

    def get_access_log(self, credential_name: Optional[str] = None) -> list[Dict[str, Any]]:
        """Get access log, optionally filtered by credential name."""
        if credential_name:
            return [e for e in self._access_log if e['credential'] == credential_name]
        return self._access_log.copy()

    def rotate_encryption(self, new_encryption_manager: EncryptionManager) -> None:
        """
        Re-encrypt all credentials with a new encryption key.

        Used during key rotation.
        """
        # Decrypt all values with old key
        decrypted = {}
        for name, cred in self._credentials.items():
            try:
                decrypted[name] = {
                    'value': self._encryption.decrypt(cred.encrypted_value),
                    'expires_at': cred.expires_at,
                    'metadata': cred.metadata,
                }
            except EncryptionError:
                # Skip credentials we can't decrypt
                pass

        # Switch to new encryption manager
        self._encryption = new_encryption_manager

        # Re-encrypt all values
        self._credentials.clear()
        for name, data in decrypted.items():
            self.store(
                name=name,
                value=data['value'],
                expires_at=data['expires_at'],
                metadata=data['metadata'],
            )


class MaskedSecret:
    """
    A wrapper that prevents secrets from being logged.

    Use this when passing secrets around to ensure they
    don't accidentally appear in logs or error messages.
    """

    def __init__(self, value: str):
        self._value = value

    def get_value(self) -> str:
        """Explicitly retrieve the secret value."""
        return self._value

    def __str__(self) -> str:
        return "********"

    def __repr__(self) -> str:
        return "MaskedSecret(********)"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MaskedSecret):
            return self._value == other._value
        return False
