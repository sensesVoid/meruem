import keyring

SERVICE = "meruem"


class VaultService:
    """Encrypted credential storage backed by the OS keychain via keyring."""

    @staticmethod
    def save(namespace: str, username: str, secret: str) -> None:
        keyring.set_password(SERVICE, f"{namespace}:{username}", secret)

    @staticmethod
    def get(namespace: str, username: str) -> str | None:
        return keyring.get_password(SERVICE, f"{namespace}:{username}")

    @staticmethod
    def delete(namespace: str, username: str) -> None:
        try:
            keyring.delete_password(SERVICE, f"{namespace}:{username}")
        except keyring.errors.PasswordDeleteError:
            pass

    @staticmethod
    def has(namespace: str, username: str) -> bool:
        return VaultService.get(namespace, username) is not None

    @staticmethod
    def redact(secret: str | None, keep: int = 4) -> str:
        """Mask a secret for display: reveal only the first and last ``keep``
        characters, filling the middle with bullets. Empty when nothing is
        stored, so the UI can distinguish "no key" from "key stored".
        """
        if not secret:
            return ""
        if len(secret) <= keep * 2:
            return "•" * len(secret)
        return f"{secret[:keep]}{'•' * 6}{secret[-keep:]}"
