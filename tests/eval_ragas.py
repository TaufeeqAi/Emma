import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root: python tests/eval_ragas.py
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.WARNING,  # Suppress noisy INFO logs during eval
    format="%(levelname)s | %(name)s | %(message)s",
)

# Suppress verbose ragas/openai/httpx logs
for noisy_logger in ["ragas", "openai", "httpx", "langfuse", "sentence_transformers"]:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EMMA Clone — RAGAS + DeepEval Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tenant",
        type=str,
        default=None,
        choices=["surgery_greenfield", "surgery_riverside"],
        help="Evaluate a single tenant. Default: all tenants.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path. Default: eval_results/ragas_<timestamp>.json",
    )
    parser.add_argument(
        "--safety-only",
        action="store_true",
        help="Run safety audit only — skip RAGAS (no OpenAI needed).",
    )
    parser.add_argument(
        "--use-groq",
        action="store_true",
        help="Use Groq Llama 3.3 as judge LLM instead of OpenAI (free, less accurate).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any threshold fails. Use in CI/CD.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-case results.",
    )
    return parser.parse_args()


def print_ragas_table(scores) -> None:
    """Print RAGAS scores as a formatted table."""
    try:
        from tabulate import tabulate  # noqa: PLC0415
    except ImportError:
        print("(install tabulate for formatted output: pip install tabulate)")
        return

    from backend.config import (  # noqa: PLC0415
        RAGAS_ANSWER_RELEVANCY_THRESHOLD,
        RAGAS_CONTEXT_PRECISION_THRESHOLD,
        RAGAS_CONTEXT_RECALL_THRESHOLD,
        RAGAS_FAITHFULNESS_THRESHOLD,
    )

    thresholds = {
        "Faithfulness":       RAGAS_FAITHFULNESS_THRESHOLD,
        "Answer Relevancy":   RAGAS_ANSWER_RELEVANCY_THRESHOLD,
        "Context Precision":  RAGAS_CONTEXT_PRECISION_THRESHOLD,
        "Context Recall":     RAGAS_CONTEXT_RECALL_THRESHOLD,
    }

    def row(label, metrics, threshold_key):
        if not metrics:
            return [label, "—", "—", "—"]
        score = getattr(metrics, threshold_key.lower().replace(" ", "_"), 0.0)
        threshold = thresholds[label]
        status = "✓" if score >= threshold else "✗"
        return [label, f"{score:.3f}", f"{threshold:.2f}", status]

    rows = [
        row("Faithfulness",      scores.aggregate, "faithfulness"),
        row("Answer Relevancy",  scores.aggregate, "answer_relevancy"),
        row("Context Precision", scores.aggregate, "context_precision"),
        row("Context Recall",    scores.aggregate, "context_recall"),
    ]

    print("\n" + "=" * 60)
    print("  RAGAS Evaluation Results — Aggregate")
    print("=" * 60)
    print(tabulate(
        rows,
        headers=["Metric", "Score", "Threshold", "Status"],
        tablefmt="rounded_outline",
    ))

    if scores.greenfield:
        print("\n  Surgery Greenfield:")
        gf_rows = [
            row("Faithfulness",      scores.greenfield, "faithfulness"),
            row("Answer Relevancy",  scores.greenfield, "answer_relevancy"),
        ]
        print(tabulate(gf_rows, headers=["Metric", "Score", "Threshold", "Status"], tablefmt="simple"))

    if scores.riverside:
        print("\n  Surgery Riverside:")
        rv_rows = [
            row("Faithfulness",      scores.riverside, "faithfulness"),
            row("Answer Relevancy",  scores.riverside, "answer_relevancy"),
        ]
        print(tabulate(rv_rows, headers=["Metric", "Score", "Threshold", "Status"], tablefmt="simple"))

    if scores.aggregate:
        status = "✓ PASSED" if scores.aggregate.passed else "✗ FAILED"
        print(f"\n  Overall: {status}")


def print_safety_table(result) -> None:
    """Print safety audit results."""
    print("\n" + "=" * 60)
    print("  Safety Audit Results")
    print("=" * 60)
    print(f"  Emergency cases tested: {result.total_emergency_cases}")
    print(f"  Correctly escalated:    {result.correctly_escalated}")
    print(f"  Missed escalations:     {len(result.missed_escalations)}")
    print(f"  Recall:                 {result.recall:.3f} ({'✓ PASS' if result.recall == 1.0 else '✗ FAIL'})")
    print(f"  Precision:              {result.precision:.3f}")
    print(f"  False positives:        {len(result.false_positives)}")

    if result.missed_escalations:
        print("\n  ⚠ MISSED ESCALATIONS (safety-critical):")
        for m in result.missed_escalations:
            print(f"    - [{m['category']}] {m['query']}")

    if result.false_positives:
        print("\n  ⚠ False positives (admin queries escalated):")
        for fp in result.false_positives:
            print(f"    - [{fp['category']}] {fp['query']}")


def main() -> int:
    """
    Main evaluation runner. Returns exit code.
    """
    args = parse_args()

    print("\n" + "=" * 60)
    print("  EMMA Clone — Evaluation Suite")
    print(f"  Timestamp: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    if args.tenant:
        print(f"  Tenant filter: {args.tenant}")
    print("=" * 60)

    exit_code = 0
    all_results = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "args": vars(args),
        "ragas": None,
        "safety": None,
        "overall_passed": True,
    }

    # ── Safety Audit 
    print("\n[1/2] Running safety audit...")
    try:
        from backend.observability.deepeval_suite import EMMADeepEvalSuite  # noqa: PLC0415
        from backend.observability.metrics import EMERGENCY_SAFETY_CASES, NON_EMERGENCY_CASES  # noqa: PLC0415

        suite = EMMADeepEvalSuite(
            tenant_id=args.tenant,
            verbose=args.verbose,
            push_to_langfuse=True,
        )
        safety_result = suite.run_safety_audit(
            emergency_cases=EMERGENCY_SAFETY_CASES,
            non_emergency_cases=NON_EMERGENCY_CASES,
            tenant_id=args.tenant or "surgery_greenfield",
        )
        print_safety_table(safety_result)
        all_results["safety"] = safety_result.to_dict()

        if not safety_result.passed:
            print("\n  ✗ SAFETY AUDIT FAILED — missed emergency escalations.")
            all_results["overall_passed"] = False
            if args.strict:
                exit_code = 1

    except Exception as exc:
        print(f"\n  ✗ Safety audit error: {exc}")
        all_results["safety"] = {"error": str(exc)}
        all_results["overall_passed"] = False
        if args.strict:
            exit_code = 1

    # ── RAGAS Evaluation 
    if not args.safety_only:
        print("\n[2/2] Running RAGAS evaluation...")
        try:
            from backend.observability.ragas_eval import RAGASEvaluator  # noqa: PLC0415

            evaluator = RAGASEvaluator(
                use_groq=args.use_groq,
                verbose=args.verbose,
            )
            scores = evaluator.run_evaluation(tenant_id=args.tenant)
            print_ragas_table(scores)
            all_results["ragas"] = scores.to_dict()

            # Save to file
            output_path = Path(args.output) if args.output else None
            saved = evaluator.save_results(scores, output_path)
            print(f"\n  Results saved to: {saved}")

            if not scores.aggregate.passed:
                all_results["overall_passed"] = False
                if args.strict:
                    print("\n  ✗ RAGAS thresholds not met — strict mode: exit 1")
                    exit_code = 1

        except ImportError as exc:
            print(
                f"\n  ⚠ RAGAS/OpenAI not installed — skipping. ({exc})\n"
                "  Run: pip install ragas==0.1.21 openai==1.54.3\n"
                "  Or:  python tests/eval_ragas.py --use-groq"
            )
            all_results["ragas"] = {"error": "dependencies not installed"}
        except Exception as exc:
            print(f"\n  ✗ RAGAS evaluation error: {exc}")
            all_results["ragas"] = {"error": str(exc)}
            all_results["overall_passed"] = False
            if args.strict:
                exit_code = 1
    else:
        print("\n[2/2] RAGAS skipped (--safety-only).")

    # ── Summary 
    overall = "✓ PASSED" if all_results["overall_passed"] else "✗ FAILED"
    print(f"\n{'='*60}")
    print(f"  Overall: {overall}")
    if args.strict and exit_code != 0:
        print("  Exiting with code 1 (--strict mode)")
    print(f"{'='*60}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())