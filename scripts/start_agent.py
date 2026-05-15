"""
Start the EMMA LiveKit agent worker process.

This is the production entry point for the agent. It runs as a long-lived
process, separate from the FastAPI server.

Usage:
    python scripts/start_agent.py
    python scripts/start_agent.py --verbose
    python scripts/start_agent.py --check   # Test LiveKit connectivity only

The agent worker:
  1. Connects to the LiveKit server.
  2. Registers as an "emma-voice-agent" worker.
  3. Waits for job dispatch (one job per incoming SIP call).
  4. Runs the EMMA voice pipeline for each call.

For horizontal scaling:
  Run multiple instances of this script on separate hosts/containers.
  LiveKit distributes jobs across all connected workers.
"""

import argparse
import asyncio
import logging
import sys

from pathlib import Path
# Add project root to Python path so 'backend' module is resolvable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("start_agent")


async def check_connectivity() -> bool:
    """Verify LiveKit server is reachable and credentials are valid."""
    from livekit import api as lk_api
    from livekit.protocol import room as room_proto
    from backend.config import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    try:
        async with lk_api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET) as client:
            resp = await client.room.list_rooms(room_proto.ListRoomsRequest())
            logger.info(
                "LiveKit connectivity OK | url=%s | active_rooms=%d",
                LIVEKIT_URL, len(resp.rooms),
            )
            return True
    except Exception as exc:
        logger.error("LiveKit connectivity failed: %s", exc)
        return False


async def main(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    load_dotenv()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.check:
        ok = await check_connectivity()
        print("✓ LiveKit connected." if ok else "✗ LiveKit connection failed.")
        return 0 if ok else 1

    logger.info("Starting EMMA LiveKit agent worker...")
    ok = await check_connectivity()
    if not ok:
        logger.error("Cannot connect to LiveKit — is livekit server running?")
        return 1

    from backend.telephony.livekit_agent import run_worker_async
    await run_worker_async()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMMA LiveKit agent worker")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--check", action="store_true", help="Connectivity check only")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    sys.exit(asyncio.run(main(args)))