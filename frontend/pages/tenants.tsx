import useSWR from "swr";
import type { NextPage } from "next";
import type {
  TenantSummary,
  SIPResourcesResponse,
  HealthResponse,
} from "../lib/types";
import { api } from "../lib/api";
import Layout from "../components/Layout";
import StatusBadge from "../components/StatusBadge";

const TenantsPage: NextPage = () => {
  const { data: tenants, error: tenantsError } = useSWR<TenantSummary[]>(
    "tenants_list",
    () => api.tenants(),
    { refreshInterval: 30_000 }
  );

  const { data: sipResources, error: sipError, mutate: mutateSip } =
    useSWR<SIPResourcesResponse>(
      "sip_resources",
      () => api.sipResources(),
      { refreshInterval: 30_000 }
    );

  const { data: ddiRouting } = useSWR(
    "ddi_routing",
    () => api.ddiRouting(),
    { refreshInterval: 30_000 }
  );

  const { data: health } = useSWR<HealthResponse>(
    "health_tenants",
    () => api.health(),
    { refreshInterval: 5_000 }
  );

  return (
    <Layout title="Multi-Tenant Management">
      {/* System health strip */}
      <div className="flex gap-2 mb-6">
        <StatusBadge
          label={health?.stt_ready ? "STT Ready" : "STT Unavailable"}
          variant={health?.stt_ready ? "green" : "red"}
          dot
        />
        <StatusBadge
          label={health?.tts_ready ? "TTS Ready" : "TTS Unavailable"}
          variant={health?.tts_ready ? "green" : "red"}
          dot
        />
        <StatusBadge
          label={`${health?.active_calls ?? 0} Active Call(s)`}
          variant={(health?.active_calls ?? 0) > 0 ? "yellow" : "grey"}
        />
      </div>

      <div className="grid grid-cols-12 gap-6">

        {/* Tenant Cards */}
        <div className="col-span-12 lg:col-span-5">
          <h2 className="text-gray-700 font-semibold mb-3">Configured Surgeries</h2>

          {tenantsError && (
            <p className="text-red-500 text-sm">Failed to load tenants.</p>
          )}

          <div className="space-y-4">
            {(tenants ?? []).map((t) => (
              <div
                key={t.tenant_id}
                className="bg-white rounded-lg border border-gray-200 shadow-sm p-4"
              >
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <h3 className="font-semibold text-gray-800">{t.surgery_name}</h3>
                    <p className="text-xs font-mono text-gray-400 mt-0.5">
                      {t.tenant_id}
                    </p>
                  </div>
                  <StatusBadge label="Active" variant="green" dot />
                </div>

                {t.phone && (
                  <div className="text-sm text-gray-600 mb-1">
                    📞 <span className="font-mono">{t.phone}</span>
                  </div>
                )}

                {t.opening_hours && (
                  <div className="mt-2 text-xs text-gray-500 space-y-0.5">
                    <p className="font-medium text-gray-600 mb-1">Opening Hours</p>
                    {Object.entries(t.opening_hours).map(([day, hours]) => (
                      <div key={day} className="flex justify-between">
                        <span className="capitalize">{day}</span>
                        <span>{hours as string}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {!tenants && !tenantsError && (
              <div className="bg-white rounded-lg border border-gray-200 p-8 text-center text-gray-400 text-sm animate-pulse">
                Loading tenants…
              </div>
            )}
          </div>
        </div>

        {/* SIP Resources */}
        <div className="col-span-12 lg:col-span-7 space-y-6">

          {/* SIP Trunks */}
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
            <div className="px-4 py-3 border-b border-gray-200 flex justify-between items-center">
              <h2 className="font-semibold text-gray-700">LiveKit SIP Trunks</h2>
              <button
                onClick={() => mutateSip()}
                className="text-xs text-nhs-blue hover:underline"
              >
                Refresh
              </button>
            </div>

            {sipError ? (
              <p className="p-4 text-red-500 text-sm">
                Cannot reach LiveKit API. Is LiveKit running?
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left text-xs text-gray-500 font-medium">Name</th>
                      <th className="px-4 py-2 text-left text-xs text-gray-500 font-medium">ID</th>
                      <th className="px-4 py-2 text-left text-xs text-gray-500 font-medium">DDI Numbers</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {sipResources?.trunks?.length === 0 ? (
                      <tr>
                        <td colSpan={3} className="px-4 py-4 text-center text-gray-400">
                          No trunks provisioned.{" "}
                          <code className="text-xs bg-gray-100 px-1 rounded">
                            python scripts/provision_sip.py
                          </code>
                        </td>
                      </tr>
                    ) : (
                      sipResources?.trunks?.map((trunk) => (
                        <tr key={trunk.id} className="hover:bg-gray-50">
                          <td className="px-4 py-2 font-medium text-gray-700">{trunk.name}</td>
                          <td className="px-4 py-2 font-mono text-xs text-gray-400">{trunk.id.slice(0, 16)}…</td>
                          <td className="px-4 py-2 text-xs text-gray-600">
                            {trunk.numbers?.length > 0
                              ? trunk.numbers.join(", ")
                              : <span className="text-gray-400">—</span>}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Dispatch Rules */}
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
            <div className="px-4 py-3 border-b border-gray-200">
              <h2 className="font-semibold text-gray-700">SIP Dispatch Rules</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs text-gray-500 font-medium">Name</th>
                    <th className="px-4 py-2 text-left text-xs text-gray-500 font-medium">ID</th>
                    <th className="px-4 py-2 text-left text-xs text-gray-500 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {sipResources?.dispatch_rules?.map((rule) => (
                    <tr key={rule.id} className="hover:bg-gray-50">
                      <td className="px-4 py-2 font-medium text-gray-700">{rule.name}</td>
                      <td className="px-4 py-2 font-mono text-xs text-gray-400">{rule.id.slice(0, 16)}…</td>
                      <td className="px-4 py-2">
                        <StatusBadge label="Active" variant="green" dot />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* DDI Routing Table */}
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
            <div className="px-4 py-3 border-b border-gray-200">
              <h2 className="font-semibold text-gray-700">DDI Routing Table</h2>
            </div>
            <div className="overflow-x-auto max-h-48">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs text-gray-500 font-medium">Number</th>
                    <th className="px-4 py-2 text-left text-xs text-gray-500 font-medium">Tenant</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {ddiRouting?.routes &&
                    Object.entries(ddiRouting.routes).map(([number, tenant]) => (
                      <tr key={number} className="hover:bg-gray-50">
                        <td className="px-4 py-2 font-mono text-xs text-gray-600">{number}</td>
                        <td className="px-4 py-2 text-xs text-gray-700">{tenant}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      </div>
    </Layout>
  );
};

export default TenantsPage;