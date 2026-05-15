import { useState, useEffect } from "react";
import useSWR from "swr";
import type { NextPage } from "next";
import type { EvalResults } from "../lib/types";
import { api } from "../lib/api";
import Layout from "../components/Layout";
import StatusBadge from "../components/StatusBadge";

function MetricBar({
  label,
  value,
  threshold,
}: {
  label:     string;
  value:     number | undefined | null;
  threshold: number;
}) {
  if (value === undefined || value === null) return null;
  const pct    = Math.round(value * 100);
  const pass   = value >= threshold;
  const width  = `${pct}%`;

  return (
    <div className="mb-4">
      <div className="flex justify-between text-sm mb-1">
        <span className="font-medium text-gray-700">{label}</span>
        <div className="flex items-center gap-2">
          <span className={`font-bold ${pass ? "text-green-700" : "text-red-700"}`}>
            {pct}%
          </span>
          <StatusBadge
            label={pass ? "PASS" : "FAIL"}
            variant={pass ? "green" : "red"}
          />
        </div>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2.5">
        <div
          className={`h-2.5 rounded-full transition-all duration-500 ${
            pass ? "bg-green-500" : "bg-red-500"
          }`}
          style={{ width }}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-400 mt-0.5">
        <span>0%</span>
        <span>threshold: {Math.round(threshold * 100)}%</span>
        <span>100%</span>
      </div>
    </div>
  );
}

const THRESHOLDS = {
  faithfulness:      0.80,
  answer_relevancy:  0.75,
  context_precision: 0.70,
  context_recall:    0.70,
};

const EvalsPage: NextPage = () => {
  const [selectedTenant, setSelectedTenant] = useState("surgery_greenfield");
  const [runningJobId, setRunningJobId] = useState<string | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);

  const { data: evals, mutate: mutateEvals, isLoading } = useSWR<EvalResults>(
    "evals_latest",
    () => api.evalsLatest(),
    { refreshInterval: runningJobId ? 2_000 : 30_000 }
  );

  // Poll job status
  const { data: jobStatus } = useSWR(
    runningJobId ? `eval_status_${runningJobId}` : null,
    () => (runningJobId ? api.evalsStatus(runningJobId) : null),
    { refreshInterval: 2_000 }
  );

  useEffect(() => {
    if (jobStatus?.status === "completed" || jobStatus?.status === "failed") {
      setRunningJobId(null);
      mutateEvals();
    }
  }, [jobStatus, mutateEvals]);

  const handleRunEvals = async () => {
    setJobError(null);
    try {
      const { job_id } = await api.evalsRun(selectedTenant);
      setRunningJobId(job_id);
    } catch (e) {
      setJobError(String(e));
    }
  };

  const ragas    = evals?.ragas;
  const deepeval = evals?.deepeval;

  return (
    <Layout title="Evaluation Dashboard">

      {/* Run controls */}
      <div className="flex flex-wrap items-center gap-3 mb-6">
        <select
          value={selectedTenant}
          onChange={(e) => setSelectedTenant(e.target.value)}
          className="text-sm border border-gray-300 rounded px-2 py-1.5 bg-white text-gray-800"
        >
          <option value="surgery_greenfield">Greenfield Medical Centre</option>
          <option value="surgery_riverside">Riverside Surgery</option>
        </select>

        <button
          onClick={handleRunEvals}
          disabled={!!runningJobId}
          className={`flex items-center gap-2 px-4 py-2 rounded text-sm font-medium transition-colors
            ${runningJobId
              ? "bg-gray-200 text-gray-500 cursor-not-allowed"
              : "bg-nhs-blue text-white hover:bg-nhs-blue-mid"}`}
        >
          {runningJobId ? (
            <>
              <span className="animate-spin">⟳</span>
              Running RAGAS + DeepEval… ({jobStatus?.stage ?? "starting"})
            </>
          ) : (
            "▶ Run Full Eval Suite"
          )}
        </button>

        {evals?.run_at && (
          <span className="text-xs text-gray-400">
            Last run: {new Date(evals.run_at).toLocaleString()}
            {evals.duration_seconds !== null && ` (${evals.duration_seconds}s)`}
          </span>
        )}

        {jobError && (
          <span className="text-xs text-red-600">Error: {jobError}</span>
        )}

        {evals?.status === "no_results" && (
          <StatusBadge label="No results yet" variant="yellow" />
        )}
      </div>

      {isLoading && (
        <p className="text-gray-400 animate-pulse mb-4">Loading eval results…</p>
      )}

      <div className="grid grid-cols-12 gap-6">

        {/* RAGAS Scores */}
        <div className="col-span-12 lg:col-span-6">
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold text-gray-800">RAGAS Evaluation</h2>
              {ragas && (
                <StatusBadge
                  label={ragas.pass ? "ALL PASS" : "FAILING"}
                  variant={ragas.pass ? "green" : "red"}
                />
              )}
            </div>

            {!ragas ? (
              <p className="text-gray-400 text-sm">No RAGAS results. Run evaluation.</p>
            ) : (
              <>
                <MetricBar
                  label="Faithfulness"
                  value={ragas.metrics.faithfulness}
                  threshold={THRESHOLDS.faithfulness}
                />
                <MetricBar
                  label="Answer Relevancy"
                  value={ragas.metrics.answer_relevancy}
                  threshold={THRESHOLDS.answer_relevancy}
                />
                <MetricBar
                  label="Context Precision"
                  value={ragas.metrics.context_precision}
                  threshold={THRESHOLDS.context_precision}
                />
                <MetricBar
                  label="Context Recall"
                  value={ragas.metrics.context_recall}
                  threshold={THRESHOLDS.context_recall}
                />

                <div className="mt-4 pt-3 border-t border-gray-100 text-xs text-gray-400">
                  {ragas.sample_count} samples · tenant: {ragas.tenant_id}
                  {ragas.source === "mock" && (
                    <span className="ml-2 text-yellow-600">(mock data)</span>
                  )}
                </div>
              </>
            )}
          </div>
        </div>

        {/* DeepEval Safety Suite */}
        <div className="col-span-12 lg:col-span-6">
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold text-gray-800">DeepEval Safety Suite</h2>
              {deepeval && (
                <div className="flex items-center gap-2">
                  <span className="text-sm font-bold text-gray-700">
                    {deepeval.passed}/{deepeval.total_tests}
                  </span>
                  <StatusBadge
                    label={deepeval.pass_rate === 1.0 ? "100% PASS" : `${Math.round(deepeval.pass_rate * 100)}%`}
                    variant={deepeval.pass_rate === 1.0 ? "green" : "red"}
                  />
                </div>
              )}
            </div>

            {!deepeval ? (
              <p className="text-gray-400 text-sm">No DeepEval results. Run evaluation.</p>
            ) : (
              <div className="space-y-1 max-h-72 overflow-y-auto">
                {deepeval.test_cases.map((tc) => (
                  <div
                    key={tc.name}
                    className={`flex items-center gap-2 px-2 py-1.5 rounded text-xs
                      ${tc.passed ? "bg-green-50 text-green-800" : "bg-red-50 text-red-800"}`}
                  >
                    <span>{tc.passed ? "✅" : "❌"}</span>
                    <span className="font-mono">{tc.name}</span>
                  </div>
                ))}
                {deepeval.source === "mock" && (
                  <p className="text-xs text-yellow-600 mt-2">(mock data)</p>
                )}
              </div>
            )}
          </div>
        </div>

      </div>
    </Layout>
  );
};

export default EvalsPage;