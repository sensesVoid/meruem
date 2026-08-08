import { cn } from "@/lib/utils";

export function Card({
  className,
  title,
  action,
  children,
}: {
  className?: string;
  title?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className={cn("rounded-lg border border-meruem-border bg-meruem-panel/60", className)}>
      {(title || action) && (
        <div className="flex items-center justify-between border-b border-meruem-border px-3 py-2">
          {title && <h3 className="text-xs font-semibold uppercase tracking-wider text-meruem-muted">{title}</h3>}
          {action}
        </div>
      )}
      <div className="p-3">{children}</div>
    </div>
  );
}