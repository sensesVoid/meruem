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
