import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import asyncio
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("provision_sip")


async def main(args: argparse.Namespace) -> int:
    from backend.telephony.sip_provisioner import SIPProvisioner

    async with SIPProvisioner() as provisioner:
        if args.check:
            trunks = await provisioner.list_trunks()
            rules  = await provisioner.list_dispatch_rules()
            print("\n── SIP Inbound Trunks ────────────────────────────────────────")
            for t in trunks:
                print(f"  {t.name:40s}  {t.sip_trunk_id}")
                for n in t.numbers:
                    print(f"    └─ {n}")
            print("\n── SIP Dispatch Rules ────────────────────────────────────────")
            for r in rules:
                print(f"  {r.name:40s}  {r.sip_dispatch_rule_id}")
            return 0

        if args.deprovision:
            confirm = input(
                "⚠️  DESTRUCTIVE: Remove ALL EMMA SIP trunks and dispatch rules? "
                "Type 'yes' to confirm: "
            )
            if confirm.strip().lower() != "yes":
                print("Aborted.")
                return 1
            await provisioner.deprovision_all()
            print("Deprovisioning complete.")
            return 0

        # Default: provision
        results = await provisioner.provision_all()
        print("\n── Provisioning Complete ─────────────────────────────────────")
        for r in results:
            print(f"  Tenant:        {r.tenant_id}")
            print(f"  Trunk ID:      {r.trunk_id}")
            print(f"  Dispatch Rule: {r.dispatch_rule_id}")
            print(f"  DDI Numbers:   {', '.join(r.ddi_numbers) or '(none configured)'}")
            print()

        print("✓ SIP trunks and dispatch rules provisioned.")
        print("  Next: python scripts/start_agent.py")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMMA LiveKit SIP provisioning")
    parser.add_argument("--check",       action="store_true", help="List existing resources")
    parser.add_argument("--deprovision", action="store_true", help="Remove all resources (DESTRUCTIVE)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)