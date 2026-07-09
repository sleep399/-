import base64
import hashlib
import json
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from app.config import settings


def _get_aes_key() -> bytes:
    key_hex = settings.aes_key
    if len(key_hex) == 64:
        return bytes.fromhex(key_hex)
    return hashlib.sha256(key_hex.encode()).digest()


def encrypt_text(plain: str) -> str:
    key = _get_aes_key()
    aesgcm = AESGCM(key)
    nonce = hashlib.md5(plain.encode()).digest()[:12]
    ct = aesgcm.encrypt(nonce, plain.encode("utf-8"), None)
    return base64.b64encode(nonce + ct).decode("utf-8")


def decrypt_text(cipher: str) -> str:
    key = _get_aes_key()
    raw = base64.b64decode(cipher.encode("utf-8"))
    nonce, ct = raw[:12], raw[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct, None).decode("utf-8")


def encrypt_json(data: dict) -> str:
    return encrypt_text(json.dumps(data, ensure_ascii=False))


def decrypt_json(cipher: str) -> dict:
    return json.loads(decrypt_text(cipher))
