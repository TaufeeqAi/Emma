import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.agents.graph import emma_graph
from backend.agents.state import make_initial_state
from backend.config import EVAL_RESULTS_DIR
from backend.observability.langfuse_client import get_langfuse_client
from backend.observability.metrics import (
    EVAL_TEST_CASES,
    EvalMetrics,
    EvalThresholds,
    PerTenantScores,
)

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    Runs the RAGAS evaluation pipeline against EMMA's LangGraph pipeline.

    Usage:
        evaluator = RAGASEvaluator()
        result = evaluator.run_evaluation()        # all tenants
        result = evaluator.run_evaluation(tenant_id="surgery_greenfield")
        evaluator.save_results(result, Path("eval_results/run_001.json"))

    Args:
        test_cases:  Override default EVAL_TEST_CASES (useful for testing).
        thresholds:  Override default score thresholds.
        use_groq:    If True, use Groq (free) instead of OpenAI as judge LLM.
                     Groq is less accurate for faithfulness scoring but costs $0.
        verbose:     If True, print per-case results as they're computed.
    """

    def __init__(
        self,
        test_cases: Optional[list[dict]] = None,
        thresholds: Optional[EvalThresholds] = None,
        use_groq: bool = False,
        verbose: bool = True,
    ) -> None:
        self._test_cases = test_cases or EVAL_TEST_CASES
        self._thresholds = thresholds or EvalThresholds()
        self._use_groq = use_groq
        self._verbose = verbose
        self._langfuse = get_langfuse_client()

    def run_evaluation(
        self,
        tenant_id: Optional[str] = None,
    ) -> PerTenantScores:
        """
        Run full RAGAS evaluation. Synchronous entry point.

        Args:
            tenant_id: If set, evaluate only this tenant. Otherwise: all tenants.

        Returns:
            PerTenantScores with per-tenant and aggregate EvalMetrics.
        """
        cases = self._test_cases
        if tenant_id:
            cases = [c for c in cases if c["tenant_id"] == tenant_id]
            if not cases:
                raise ValueError(
                    f"No test cases found for tenant_id='{tenant_id}'. "
                    f"Valid values: {list({c['tenant_id'] for c in self._test_cases})}"
                )

        logger.info(
            "Starting RAGAS evaluation | %d cases | tenant_filter=%s",
            len(cases), tenant_id or "all",
        )

        # Step 1: Collect pipeline outputs for all cases
        pipeline_outputs = self._collect_pipeline_outputs(cases)

        # Step 2: Compute RAGAS scores (may involve LLM calls)
        scores = PerTenantScores()

        # Aggregate across all cases
        aggregate_metrics = self._compute_ragas(pipeline_outputs, tenant_id="all")
        scores.aggregate = aggregate_metrics

        # Per-tenant breakdown (only if running across all tenants)
        if not tenant_id:
            greenfield_outputs = [
                o for o in pipeline_outputs
                if o["tenant_id"] == "surgery_greenfield"
            ]
            riverside_outputs = [
                o for o in pipeline_outputs
                if o["tenant_id"] == "surgery_riverside"
            ]
            if greenfield_outputs:
                scores.greenfield = self._compute_ragas(
                    greenfield_outputs, tenant_id="surgery_greenfield"
                )
            if riverside_outputs:
                scores.riverside = self._compute_ragas(
                    riverside_outputs, tenant_id="surgery_riverside"
                )

        # Step 3: Push scores to Langfuse (if enabled)
        if self._langfuse and scores.aggregate:
            run_trace_id = str(uuid.uuid4())
            self._langfuse.create_trace(
                trace_id=run_trace_id,
                name="ragas_evaluation_run",
                query=f"RAGAS eval | {len(cases)} cases",
                tenant_id=tenant_id or "all",
                metadata={"n_cases": len(cases)},
            )
            self._langfuse.score_turn(
                trace_id=run_trace_id,
                metrics=scores.aggregate,
                comment=f"Automated RAGAS evaluation — {len(cases)} cases",
            )
            self._langfuse.flush()

        return scores

    def save_results(
        self,
        scores: PerTenantScores,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Save evaluation results to a JSON file.

        Files are named by timestamp if no path is provided.
        Accumulates a history of runs for trend analysis.

        Args:
            scores:      PerTenantScores from run_evaluation().
            output_path: Override output file path.

        Returns:
            Path to the saved JSON file.
        """
        if output_path is None:
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_path = EVAL_RESULTS_DIR / f"ragas_{ts}.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "run_id": str(uuid.uuid4()),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "scores": scores.to_dict(),
            "thresholds": {
                "faithfulness": self._thresholds.faithfulness,
                "answer_relevancy": self._thresholds.answer_relevancy,
                "context_precision": self._thresholds.context_precision,
                "context_recall": self._thresholds.context_recall,
            },
            "passed": scores.aggregate.passed if scores.aggregate else False,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("RAGAS results saved to: %s", output_path)
        return output_path

    def assert_thresholds(self, scores: PerTenantScores) -> None:
        """
        Assert all aggregate scores meet configured thresholds.
        Raises AssertionError with a descriptive message if any score fails.
        Used in CI/CD pipelines to gate releases.

        Args:
            scores: PerTenantScores from run_evaluation().

        Raises:
            AssertionError: if any score is below its threshold.
            ValueError: if aggregate scores are None (evaluation failed).
        """
        if scores.aggregate is None:
            raise ValueError("No aggregate scores available — evaluation may have failed.")

        m = scores.aggregate
        t = self._thresholds

        failures = []
        if m.faithfulness < t.faithfulness:
            failures.append(
                f"faithfulness {m.faithfulness:.3f} < threshold {t.faithfulness}"
            )
        if m.answer_relevancy < t.answer_relevancy:
            failures.append(
                f"answer_relevancy {m.answer_relevancy:.3f} < threshold {t.answer_relevancy}"
            )
        if m.context_precision < t.context_precision:
            failures.append(
                f"context_precision {m.context_precision:.3f} < threshold {t.context_precision}"
            )
        if m.context_recall < t.context_recall:
            failures.append(
                f"context_recall {m.context_recall:.3f} < threshold {t.context_recall}"
            )

        if failures:
            raise AssertionError(
                "RAGAS thresholds not met:\n" + "\n".join(f"  ✗ {f}" for f in failures)
            )

        logger.info("✓ All RAGAS thresholds passed.")

    # ── Private methods ────────────────────────────────────────────────────────

    def _collect_pipeline_outputs(self, cases: list[dict]) -> list[dict]:
        """
        Run each test case through the EMMA pipeline and collect outputs.

        Returns:
            List of dicts:
              {
                question, answer, contexts (list[str]),
                ground_truth, tenant_id, category,
                escalated, verified, error
              }
        """
        outputs = []
        for i, case in enumerate(cases):
            question = case["question"]
            tid = case["tenant_id"]
            ground_truth = case["ground_truth"]
            category = case.get("category", "unknown")

            if self._verbose:
                print(f"  [{i+1}/{len(cases)}] {tid} | {question[:60]}...")

            try:
                state = make_initial_state(query=question, tenant_id=tid)
                result = emma_graph.invoke(state)

                answer = result.get("final_response") or ""
                chunks = result.get("retrieved_chunks") or []
                contexts = [c["text"] for c in chunks] if chunks else [""]

                output = {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                    "tenant_id": tid,
                    "category": category,
                    "escalated": result.get("escalate", False),
                    "verified": result.get("verified", False),
                    "error": result.get("error"),
                }
                outputs.append(output)

                if self._verbose:
                    print(f"       answer={answer[:80]}...")

            except Exception as exc:
                logger.error(
                    "Pipeline failed for case '%s' (tenant=%s): %s",
                    question, tid, exc,
                )
                # Include failed case with empty answer — will score low
                outputs.append({
                    "question": question,
                    "answer": "",
                    "contexts": [""],
                    "ground_truth": ground_truth,
                    "tenant_id": tid,
                    "category": category,
                    "escalated": False,
                    "verified": False,
                    "error": str(exc),
                })

        return outputs

    def _compute_ragas(
        self,
        outputs: list[dict],
        tenant_id: str,
    ) -> EvalMetrics:
        """
        Compute RAGAS scores for a list of pipeline outputs.

        Args:
            outputs:   List from _collect_pipeline_outputs().
            tenant_id: Label for this metric set ("all" or specific tenant).

        Returns:
            EvalMetrics with computed scores. On error: returns zeros with error field.
        """
        try:
            from ragas import evaluate  # noqa: PLC0415
            from ragas.metrics import (  # noqa: PLC0415
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
            from datasets import Dataset  # noqa: PLC0415
        except ImportError as exc:
            logger.error("RAGAS dependencies not installed: %s", exc)
            return EvalMetrics(0.0, 0.0, 0.0, 0.0, error=str(exc), tenant_id=tenant_id)

        # RAGAS requires non-empty contexts — filter escalated cases
        # (escalated cases have no RAG context and would skew scores)
        eval_outputs = [o for o in outputs if not o["escalated"] and o["answer"]]
        if not eval_outputs:
            logger.warning(
                "No non-escalated outputs for RAGAS evaluation (tenant=%s)", tenant_id
            )
            return EvalMetrics(0.0, 0.0, 0.0, 0.0, tenant_id=tenant_id,
                               error="No non-escalated cases available")

        dataset = Dataset.from_list([
            {
                "question":     o["question"],
                "answer":       o["answer"],
                "contexts":     o["contexts"],
                "ground_truth": o["ground_truth"],
            }
            for o in eval_outputs
        ])

        # Configure judge LLM
        llm_config = self._get_llm_config()

        logger.info(
            "Computing RAGAS scores | %d cases | tenant=%s | judge=%s",
            len(eval_outputs), tenant_id,
            "groq" if self._use_groq else "openai",
        )

        try:
            metrics_list = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
            if llm_config:
                for metric in metrics_list:
                    if hasattr(metric, "llm"):
                        metric.llm = llm_config

            scores = evaluate(
                dataset,
                metrics=metrics_list,
            )

            run_id = str(uuid.uuid4())
            result = EvalMetrics(
                faithfulness=float(scores.get("faithfulness", 0.0)),
                answer_relevancy=float(scores.get("answer_relevancy", 0.0)),
                context_precision=float(scores.get("context_precision", 0.0)),
                context_recall=float(scores.get("context_recall", 0.0)),
                run_id=run_id,
                tenant_id=tenant_id if tenant_id != "all" else None,
                n_cases=len(eval_outputs),
            )

            logger.info(
                "RAGAS | tenant=%s | faith=%.3f | relevancy=%.3f | "
                "precision=%.3f | recall=%.3f | passed=%s",
                tenant_id,
                result.faithfulness,
                result.answer_relevancy,
                result.context_precision,
                result.context_recall,
                result.passed,
            )
            return result

        except Exception as exc:
            logger.exception("RAGAS evaluate() failed (tenant=%s): %s", tenant_id, exc)
            return EvalMetrics(0.0, 0.0, 0.0, 0.0, tenant_id=tenant_id, error=str(exc))

    def _get_llm_config(self):
        """
        Return a RAGAS-compatible LLM config for the judge model.

        Groq option: free, fast, slightly less accurate for faithfulness.
        OpenAI option: costs ~$0.01 per eval run with gpt-4o-mini, more accurate.

        Returns:
            LLM config object or None (uses RAGAS default = OpenAI gpt-3.5-turbo).
        """
        if not self._use_groq:
            return None  # Use RAGAS default (OpenAI, from OPENAI_API_KEY env var)

        try:
            from langchain_groq import ChatGroq  # noqa: PLC0415
            from backend.config import GROQ_API_KEY  # noqa: PLC0415
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=GROQ_API_KEY,
                temperature=0,
            )
        except ImportError:
            logger.warning("langchain-groq not available — using RAGAS default LLM.")
            return None