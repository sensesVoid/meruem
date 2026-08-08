export function Input({
  className,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={`w-full rounded-md border border-meruem-border bg-meruem-bg px-2.5 py-1.5 text-sm text-meruem-text placeholder:text-meruem-muted focus:border-meruem-accent focus:outline-none ${className ?? ""}`}
      {...props}
    />
  );
}

export function Select({
  className,
  children,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={`w-full rounded-md border border-meruem-border bg-meruem-bg px-2.5 py-1.5 text-sm text-meruem-text focus:border-meruem-accent focus:outline-none ${className ?? ""}`}
      {...props}
    >
      {children}
    </select>
  );
}

export function Textarea({
  className,
  ...props
}: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={`w-full rounded-md border border-meruem-border bg-meruem-bg px-2.5 py-1.5 text-sm text-meruem-text placeholder:text-meruem-muted focus:border-meruem-accent focus:outline-none ${className ?? ""}`}
      {...props}
    />
  );
}