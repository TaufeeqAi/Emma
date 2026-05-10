import sys
import argparse
import logging
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import TENANTS, load_tenant_config
from backend.rag.ingestor import SurgeryIngestor

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest GP surgery guidelines into Qdrant."
    )
    parser.add_argument(
        "--tenant",
        type=str,
        default=None,
        help="Ingest a single tenant by ID. Default: all tenants.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After ingestion, verify collection vector counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tenants_to_ingest = [args.tenant] if args.tenant else TENANTS

    print("\n" + "=" * 60)
    print("  EMMA Clone — Phase 1 Ingestion Pipeline")
    print("=" * 60)
    print(f"  Tenants: {tenants_to_ingest}")
    print("=" * 60 + "\n")

    ingestor = SurgeryIngestor()
    results = []
    errors = []

    for tenant_id in tenants_to_ingest:
        # Validate config exists before attempting ingestion
        try:
            config = load_tenant_config(tenant_id)
            print(f"Surgery: {config['surgery_name']} ({tenant_id})")
        except (FileNotFoundError, ValueError) as e:
            logger.error("Config error for '%s': %s", tenant_id, e)
            errors.append({"tenant_id": tenant_id, "error": str(e)})
            continue

        try:
            result = ingestor.ingest_tenant(tenant_id)
            results.append(result)
            print(
                f"  ✓ {result['chunks_created']} chunks | "
                f"{result['vectors_stored']} vectors | "
                f"collection: {result['collection']}\n"
            )
        except Exception as e:
            logger.exception("Ingestion failed for '%s'", tenant_id)
            errors.append({"tenant_id": tenant_id, "error": str(e)})
            print(f"  ✗ FAILED: {e}\n")

    # Verification pass
    if args.verify and results:
        print("\n" + "─" * 60)
        print("  Verification:")
        for r in results:
            info = ingestor.verify_collection(r["tenant_id"])
            status = info.get("status", "unknown")
            count = info.get("vectors_count", "?")
            print(f"  {r['tenant_id']}: {count} vectors | status={status}")

    # Summary
    print("\n" + "=" * 60)
    print("  Ingestion Summary")
    print("=" * 60)
    for r in results:
        print(
            f"  ✓ {r['tenant_id']:<30} "
            f"{r['chunks_created']} chunks → {r['vectors_stored']} vectors"
        )
    for e in errors:
        print(f"  ✗ {e['tenant_id']:<30} ERROR: {e['error']}")

    if errors:
        print(f"\n  {len(errors)} tenant(s) failed. Check logs above.")
        sys.exit(1)
    else:
        print(f"\n  All {len(results)} tenant(s) ingested successfully.")
        print("  Next step: pytest tests/test_retrieval.py -v")


if __name__ == "__main__":
    main()