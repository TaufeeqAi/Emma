import json
import logging
from dataclasses import dataclass
from typing import Optional

from livekit import api as lk_api
from livekit.protocol import sip as sip_proto

from backend.config import (
    LIVEKIT_URL,
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    SIP_AUTH_USERNAME,
    SIP_AUTH_PASSWORD,
    TENANTS,
)
from backend.telephony.ddi_router import get_ddi_router

logger = logging.getLogger(__name__)


@dataclass
class ProvisionedTenant:
    """Result of provisioning a single tenant's SIP resources."""
    tenant_id: str
    trunk_id: str
    trunk_name: str
    dispatch_rule_id: str
    dispatch_rule_name: str
    ddi_numbers: list[str]


class SIPProvisioner:
    """
    Manages LiveKit SIP trunks and dispatch rules for all EMMA tenants.

    Usage:
        async with SIPProvisioner() as provisioner:
            results = await provisioner.provision_all()
            for result in results:
                print(f"{result.tenant_id}: trunk={result.trunk_id}")
    """

    def __init__(self) -> None:
        self._router = get_ddi_router()
        self._client: Optional[lk_api.LiveKitAPI] = None

    async def __aenter__(self):
        self._client = lk_api.LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
        return self

    async def __aexit__(self, *_):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def provision_all(self) -> list[ProvisionedTenant]:
        """
        Provision SIP trunks and dispatch rules for all configured tenants.
        Idempotent: skips resources that already exist (matched by name).
        """
        results = []
        for tenant_id in TENANTS:
            try:
                result = await self._provision_tenant(tenant_id)
                results.append(result)
                logger.info(
                    "Provisioned | tenant=%s | trunk=%s | rule=%s",
                    tenant_id, result.trunk_id, result.dispatch_rule_id,
                )
            except Exception as exc:
                logger.error("Provisioning failed for tenant '%s': %s", tenant_id, exc)
                raise
        return results

    async def deprovision_all(self) -> None:
        """Remove all EMMA SIP trunks and dispatch rules. Destructive."""
        logger.warning("Deprovisioning all EMMA SIP resources — this is destructive!")
        await self._delete_emma_dispatch_rules()
        await self._delete_emma_trunks()

    async def list_trunks(self) -> list:
        """List all SIP inbound trunks. Returns raw protobuf objects."""
        resp = await self._client.sip.list_sip_inbound_trunk(
            sip_proto.ListSIPInboundTrunkRequest()
        )
        return resp.items

    async def list_dispatch_rules(self) -> list:
        """List all SIP dispatch rules. Returns raw protobuf objects."""
        resp = await self._client.sip.list_sip_dispatch_rule(
            sip_proto.ListSIPDispatchRuleRequest()
        )
        return resp.items

    async def _provision_tenant(self, tenant_id: str) -> ProvisionedTenant:
        ddi_numbers = self._router.ddi_numbers_for_tenant(tenant_id)
        if not ddi_numbers:
            ddi_numbers = []
            logger.warning(
                "No DDI numbers configured for tenant '%s'. "
                "Trunk will be created without number routing.",
                tenant_id,
            )

        trunk_id = await self._ensure_trunk(tenant_id, ddi_numbers)
        rule_id = await self._ensure_dispatch_rule(tenant_id, trunk_id)

        return ProvisionedTenant(
            tenant_id=tenant_id,
            trunk_id=trunk_id,
            trunk_name=_trunk_name(tenant_id),
            dispatch_rule_id=rule_id,
            dispatch_rule_name=_rule_name(tenant_id),
            ddi_numbers=ddi_numbers,
        )

    async def _ensure_trunk(self, tenant_id: str, ddi_numbers: list[str]) -> str:
        """Create or update the SIP inbound trunk for a tenant."""
        name = _trunk_name(tenant_id)

        existing = await self._find_trunk_by_name(name)
        if existing:
            logger.debug("Trunk '%s' already exists: %s", name, existing.sip_trunk_id)
            return existing.sip_trunk_id

        trunk_req = sip_proto.CreateSIPInboundTrunkRequest(
            trunk=sip_proto.SIPInboundTrunkInfo(
                name=name,
                numbers=ddi_numbers,
                auth_username=f"{SIP_AUTH_USERNAME}-{tenant_id}",
                auth_password=SIP_AUTH_PASSWORD,
                krisp_enabled=False,
            )
        )
        trunk = await self._client.sip.create_sip_inbound_trunk(trunk_req)
        logger.info(
            "Created SIP trunk | name=%s | id=%s | numbers=%s",
            name, trunk.sip_trunk_id, ddi_numbers,
        )
        return trunk.sip_trunk_id

    async def _ensure_dispatch_rule(
        self, tenant_id: str, trunk_id: str
    ) -> str:
        """
        Create a dispatch rule that routes calls to individual rooms.

        FIX: CreateSIPDispatchRuleRequest takes:
          - rule: SIPDispatchRule (the routing logic)
          - trunk_ids: list of trunk IDs (top-level)
          - name: display name (top-level)
          - metadata: JSON room metadata (top-level)
          - attributes: key-value pairs (top-level)
        """
        name = _rule_name(tenant_id)

        existing = await self._find_rule_by_name(name)
        if existing:
            logger.debug("Dispatch rule '%s' already exists: %s", name, existing.sip_dispatch_rule_id)
            return existing.sip_dispatch_rule_id

        # Build the routing logic (SIPDispatchRule)
        dispatch_rule = sip_proto.SIPDispatchRule(
            dispatch_rule_individual=sip_proto.SIPDispatchRuleIndividual(
                room_prefix=f"{tenant_id}-",
                pin="",
            )
        )

        # Build the request with rule as a separate field
        rule_req = sip_proto.CreateSIPDispatchRuleRequest(
            rule=dispatch_rule,
            trunk_ids=[trunk_id],
            name=name,
            metadata=json.dumps({
                "tenant_id":    tenant_id,
                "emma_version": "5",
                "agent_name":   "emma-voice-agent",
            }),
            attributes={
                "tenant_id": tenant_id,
                "service":   "emma",
            },
        )
        rule = await self._client.sip.create_sip_dispatch_rule(rule_req)
        logger.info(
            "Created dispatch rule | name=%s | id=%s | trunk=%s | prefix=%s-",
            name, rule.sip_dispatch_rule_id, trunk_id, tenant_id,
        )
        return rule.sip_dispatch_rule_id

    async def _find_trunk_by_name(self, name: str) -> Optional[sip_proto.SIPInboundTrunkInfo]:
        trunks = await self.list_trunks()
        return next((t for t in trunks if t.name == name), None)

    async def _find_rule_by_name(self, name: str) -> Optional[sip_proto.SIPDispatchRuleInfo]:
        rules = await self.list_dispatch_rules()
        return next((r for r in rules if r.name == name), None)

    async def _delete_emma_trunks(self) -> None:
        trunks = await self.list_trunks()
        for trunk in trunks:
            if trunk.name.startswith("EMMA "):
                await self._client.sip.delete_sip_trunk(
                    sip_proto.DeleteSIPTrunkRequest(sip_trunk_id=trunk.sip_trunk_id)
                )
                logger.info("Deleted trunk %s (%s)", trunk.name, trunk.sip_trunk_id)

    async def _delete_emma_dispatch_rules(self) -> None:
        rules = await self.list_dispatch_rules()
        for rule in rules:
            if rule.name.startswith("EMMA "):
                await self._client.sip.delete_sip_dispatch_rule(
                    sip_proto.DeleteSIPDispatchRuleRequest(
                        sip_dispatch_rule_id=rule.sip_dispatch_rule_id
                    )
                )
                logger.info("Deleted rule %s (%s)", rule.name, rule.sip_dispatch_rule_id)

def _trunk_name(tenant_id: str) -> str:
    return f"EMMA {tenant_id.replace('_', ' ').title()} Inbound"


def _rule_name(tenant_id: str) -> str:
    return f"EMMA {tenant_id.replace('_', ' ').title()} Dispatch"