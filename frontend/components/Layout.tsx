import Link from "next/link";
import { useRouter } from "next/router";
import clsx from "clsx";
import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
  title?:   string;
}

const NAV_ITEMS = [
  { href: "/",         label: "Live Monitor" },
  { href: "/tenants",  label: "Tenants"      },
  { href: "/evals",    label: "Evaluations"  },
] as const;

export default function Layout({ children, title = "EMMA Clone" }: Props) {
  const { pathname } = useRouter();

  return (
    <div className="min-h-screen bg-nhs-grey-lt font-sans">
      {/* NHS-style header */}
      <header className="bg-nhs-blue shadow-sm">
        <div className="max-w-screen-2xl mx-auto px-4 py-3 flex items-center gap-6">
          <span className="text-white font-bold text-lg tracking-wide">
            🏥 EMMA Clone
          </span>
          <nav className="flex gap-1">
            {NAV_ITEMS.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                className={clsx(
                  "px-3 py-1.5 rounded text-sm font-medium transition-colors",
                  pathname === href
                    ? "bg-nhs-blue-lt text-white"
                    : "text-nhs-grey-lt hover:bg-nhs-blue-mid text-white"
                )}
              >
                {label}
              </Link>
            ))}
          </nav>
          <div className="ml-auto text-nhs-grey-lt text-xs opacity-75">
            NHS GP AI Receptionist
          </div>
        </div>
      </header>

      {/* Page title bar */}
      {title && (
        <div className="bg-white border-b border-gray-200 px-6 py-3">
          <h1 className="text-nhs-blue font-semibold text-xl">{title}</h1>
        </div>
      )}

      {/* Main content */}
      <main className="max-w-screen-2xl mx-auto px-4 py-6">{children}</main>
    </div>
  );
}