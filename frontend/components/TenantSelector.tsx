import useSWR from "swr";
import { api } from "../lib/api";
import type { TenantSummary } from "../lib/types";

interface Props {
  value:    string | null;
  onChange: (tenantId: string | null) => void;
}

export default function TenantSelector({ value, onChange }: Props) {
  const { data: tenants } = useSWR<TenantSummary[]>(
    "tenants",
    () => api.tenants(),
    { refreshInterval: 30_000 }
  );

  return (
    <div className="flex items-center gap-2">
      <label className="text-sm text-gray-600 font-medium whitespace-nowrap">
        Surgery:
      </label>
      <select
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value || null)}
        className="text-sm border border-gray-300 rounded px-2 py-1.5 bg-white text-gray-800 focus:outline-none focus:border-nhs-blue"
      >
        <option value="">All surgeries</option>
        {tenants?.map((t) => (
          <option key={t.tenant_id} value={t.tenant_id}>
            {t.surgery_name}
          </option>
        ))}
      </select>
    </div>
  );
}