import logging
import os
from typing import Optional

from backend.config import TENANTS, DEFAULT_TENANT, load_tenant_config

logger = logging.getLogger(__name__)

# ── Static routing table 
# Maps normalised phone number / SIP extension → tenant_id.
# Loaded from tenant config files at startup; can be updated at runtime.
#
# Format:
#   "07700000001" → "surgery_greenfield"   (UK mobile, stripped of +44 prefix)
#   "441234567890" → "surgery_greenfield"  (international format, no +)
#   "1000"         → "surgery_greenfield"  (SIP extension for dev/testing)

_STATIC_ROUTING_TABLE: dict[str, str] = {
    # Surgery Greenfield
    "01234567890":   "surgery_greenfield",
    "441234567890":  "surgery_greenfield",
    "1000":          "surgery_greenfield",  # SIP extension (dev/SIPp)
    "9000":          "surgery_greenfield",  # Alt SIP extension
    # Surgery Riverside
    "01234987654":   "surgery_riverside",
    "441234987654":  "surgery_riverside",
    "1001":          "surgery_riverside",   # SIP extension (dev/SIPp)
    "9001":          "surgery_riverside",   # Alt SIP extension
}


class DDIRouter:
    """
    Maps PSTN DDI numbers and SIP extensions to EMMA tenant_ids.

    Also provides the inverse mapping needed for SIP provisioning:
      tenant_id → list[DDI numbers] used to configure LiveKit SIP trunks.

    Thread safety: read-only after init; add_route() is not thread-safe.
    For multi-worker deployments (Phase 6), move routing table to Redis.
    """

    def __init__(self) -> None:
        self._table: dict[str, str] = dict(_STATIC_ROUTING_TABLE)
        self._load_from_tenant_configs()
        logger.info(
            "DDIRouter initialised with %d route(s): %s",
            len(self._table),
            self._table,
        )

    def route(self, dialed_number: str) -> str:
        """
        Resolve a dialled number or SIP extension to a tenant_id.

        Normalisation steps (in order):
          1. Strip whitespace, dashes, parentheses, leading +
          2. Direct table lookup
          3. Strip leading 0 and prepend 44 (UK local → international)
          4. Fall through to DEFAULT_TENANT with a warning

        Args:
            dialed_number: The destination number from SIP INVITE or LiveKit
                           SIP participant identity.

        Returns:
            tenant_id string.
        """
        normalised = self._normalise(dialed_number)

        if normalised in self._table:
            tenant = self._table[normalised]
            logger.debug("DDI exact match: '%s' → '%s'", normalised, tenant)
            return tenant

        if normalised.startswith("0"):
            intl = "44" + normalised[1:]
            if intl in self._table:
                tenant = self._table[intl]
                logger.debug("DDI intl-prefix match: '%s' → '%s'", intl, tenant)
                return tenant

        logger.warning(
            "DDIRouter: unrecognised number '%s' → default tenant '%s'. "
            "Add this number to the routing table or tenant config.",
            dialed_number, DEFAULT_TENANT,
        )
        return DEFAULT_TENANT

    def room_name_for_tenant(self, tenant_id: str, call_uid: str) -> str:
        """
        Generate a LiveKit room name for a specific call.

        Format: "{tenant_id}-{call_uid}"
        The call_uid is typically a UUID4 suffix appended by the SIP dispatch rule.

        Args:
            tenant_id: The surgery tenant identifier.
            call_uid:  Unique call identifier (uuid4).

        Returns:
            Room name string compatible with LiveKit room name constraints.
        """
        return f"{tenant_id}-{call_uid}"

    def tenant_from_room_name(self, room_name: str) -> Optional[str]:
        """
        Extract tenant_id from a LiveKit room name.

        Tries each known tenant prefix in order (longest prefix first to
        avoid prefix ambiguity if "surgery_green" and "surgery_greenfield" exist).

        Args:
            room_name: LiveKit room name, e.g. "surgery_greenfield-abc123"

        Returns:
            tenant_id string, or None if no prefix matched.
        """
        for tenant_id in sorted(TENANTS, key=len, reverse=True):
            if room_name.startswith(tenant_id + "-"):
                return tenant_id
        logger.warning("Cannot extract tenant from room name '%s'", room_name)
        return None

    def ddi_numbers_for_tenant(self, tenant_id: str) -> list[str]:
        """
        Return all DDI numbers that route to the given tenant_id.

        Used by SipProvisioner to create LiveKit SIP trunks.
        Includes both normalised and +E.164 formatted versions
        since some SIP carriers expect the +44... format.

        Args:
            tenant_id: The surgery tenant identifier.

        Returns:
            List of E.164-formatted numbers (e.g. "+441234567890").
        """
        numbers = set()
        for number, tid in self._table.items():
            if tid == tenant_id and number.isdigit() and len(number) >= 10:
                # Skip SIP extensions (short numbers < 5 digits)
                if len(number) >= 10:
                    numbers.add("+" + number.lstrip("0") if not number.startswith("44")
                                else "+" + number)
        return sorted(numbers)

    def add_route(self, number: str, tenant_id: str) -> None:
        if tenant_id not in TENANTS:
            raise ValueError(
                f"Unknown tenant_id '{tenant_id}'. Valid tenants: {TENANTS}"
            )
        normalised = self._normalise(number)
        self._table[normalised] = tenant_id
        logger.info("DDI route added: '%s' → '%s'", normalised, tenant_id)

    def remove_route(self, number: str) -> bool:
        normalised = self._normalise(number)
        if normalised in self._table:
            if self._table[normalised] == DEFAULT_TENANT:
                logger.warning("Refusing to remove default tenant route for '%s'", normalised)
                return False
            del self._table[normalised]
            logger.info("DDI route removed: '%s'", normalised)
            return True
        return False

    def get_all_routes(self) -> dict[str, str]:
        return dict(self._table)

    @property
    def default_tenant(self) -> str:
        return DEFAULT_TENANT

    def _normalise(self, number: str) -> str:
        return (
            number.strip()
            .replace(" ", "")
            .replace("-", "")
            .replace("(", "")
            .replace(")", "")
            .lstrip("+")
        )

    def _load_from_tenant_configs(self) -> None:
        for tenant_id in TENANTS:
            try:
                config = load_tenant_config(tenant_id)
                ddi = config.get("ddi_number")
                if ddi:
                    normalised = self._normalise(ddi)
                    self._table[normalised] = tenant_id
                    logger.debug("Loaded DDI from config: '%s' → '%s'", normalised, tenant_id)
            except (FileNotFoundError, ValueError) as exc:
                logger.debug("Could not load DDI config for '%s': %s", tenant_id, exc)


# ── Module-level singleton 
_ddi_router_instance: Optional[DDIRouter] = None


def get_ddi_router() -> DDIRouter:
    global _ddi_router_instance
    if _ddi_router_instance is None:
        _ddi_router_instance = DDIRouter()
    return _ddi_router_instance