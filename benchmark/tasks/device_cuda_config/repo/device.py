"""Device selection. Training can only use the devices this machine exposes."""

AVAILABLE_DEVICES = ("cpu",)


def resolve_device(name):
    if name not in AVAILABLE_DEVICES:
        raise RuntimeError(f"device {name!r} was requested but is not available (found: {', '.join(AVAILABLE_DEVICES)})")
    return name
