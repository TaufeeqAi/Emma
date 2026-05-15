import clsx from "clsx";

type Variant = "green" | "yellow" | "red" | "blue" | "grey";

interface Props {
  label:    string;
  variant?: Variant;
  dot?:     boolean;
}

const VARIANT_CLASSES: Record<Variant, string> = {
  green:  "bg-green-100 text-green-800 border-green-300",
  yellow: "bg-yellow-100 text-yellow-800 border-yellow-300",
  red:    "bg-red-100 text-red-800 border-red-300",
  blue:   "bg-nhs-blue text-white border-nhs-blue",
  grey:   "bg-gray-100 text-gray-600 border-gray-300",
};

const DOT_CLASSES: Record<Variant, string> = {
  green:  "bg-green-500",
  yellow: "bg-yellow-500",
  red:    "bg-red-500",
  blue:   "bg-nhs-blue-lt",
  grey:   "bg-gray-400",
};

export default function StatusBadge({ label, variant = "grey", dot }: Props) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border",
        VARIANT_CLASSES[variant]
      )}
    >
      {dot && (
        <span
          className={clsx("w-1.5 h-1.5 rounded-full animate-pulse", DOT_CLASSES[variant])}
        />
      )}
      {label}
    </span>
  );
}