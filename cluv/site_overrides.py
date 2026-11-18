"""Immutable input snapshot for submit-only, site-specific resource overrides."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import tomllib

RESOURCE_KEYS = {
    "nodes",
    "ntasks",
    "ntasks-per-node",
    "gpus",
    "gpus-per-node",
    "gpus-per-task",
    "cpus-per-task",
    "mem",
    "mem-per-cpu",
    "time",
    "constraint",
}
_RESOURCE_VALUE = re.compile(r"[A-Za-z0-9_.:+%/|&!*-]+")


@dataclass(frozen=True)
class SiteOverride:
    resources: tuple[dict, ...] | None = None
    program_args_prefix: tuple[str, ...] = ()


@dataclass(frozen=True)
class SiteOverrides:
    path: Path
    raw: bytes
    sha256: str
    sites: dict[str, SiteOverride]

    def for_site(self, name: str) -> SiteOverride:
        # Resource merging and variant expansion must not mutate the input snapshot.
        return deepcopy(self.sites.get(name, SiteOverride()))


def load_site_overrides(path: Path, configured_sites: list[str]) -> SiteOverrides:
    """Read exactly once. No transport or project mutation is allowed here."""
    raw = path.read_bytes()
    document = tomllib.loads(raw.decode("utf-8"))
    if set(document) != {"sites"} or not isinstance(document["sites"], dict):
        raise ValueError("Site overrides require only a [sites.NAME] table")
    sites = {}
    for name, values in document["sites"].items():
        if name not in configured_sites:
            raise ValueError(f"Unknown configured site in overrides: {name}")
        if not isinstance(values, dict) or set(values) - {"resources", "program_args_prefix"}:
            raise ValueError(f"Unsupported site override fields for {name}")
        resources = values.get("resources")
        if resources is not None:
            if not isinstance(resources, list) or not resources:
                raise ValueError(f"{name}.resources must contain one table per header component")
            for component in resources:
                if not isinstance(component, dict) or set(component) - RESOURCE_KEYS:
                    raise ValueError(f"{name}.resources contains unsupported resource keys")
                for key, value in component.items():
                    valid = (type(value) is int and value > 0) or value is False
                    valid |= isinstance(value, str) and bool(_RESOURCE_VALUE.fullmatch(value))
                    if not valid:
                        raise ValueError(f"Invalid resource value for {name}.{key}")
            resources = tuple(deepcopy(resources))
        prefix = values.get("program_args_prefix", [])
        if not isinstance(prefix, list) or any(
            not isinstance(arg, str) or arg == "--" or "\x00" in arg for arg in prefix
        ):
            raise ValueError(f"{name}.program_args_prefix must be literal strings without '--'")
        sites[name] = SiteOverride(resources, tuple(prefix))
    return SiteOverrides(path.resolve(), raw, hashlib.sha256(raw).hexdigest(), sites)
