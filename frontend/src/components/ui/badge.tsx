import { cn } from "@/lib/utils";

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: React.ReactNode;
  tone?: "neutral" | "green" | "red" | "gold" | "blue";
  className?: string;
}) {
  const tones: Record<string, string> = {
    neutral: "bg-meruem-panel text-meruem-muted border-meruem-border",
    green: "bg-meruem-green/15 text-meruem-green border-meruem-green/40",
    red: "bg-meruem-red/15 text-meruem-red border-meruem-red/40",
    gold: "bg-meruem-gold/15 text-meruem-gold border-meruem-gold/40",
    blue: "bg-meruem-accent/15 text-meruem-accent border-meruem-accent/40",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[11px] font-medium",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}