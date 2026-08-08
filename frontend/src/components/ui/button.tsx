import { cn } from "@/lib/utils";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "ghost" | "danger" | "success";
  size?: "sm" | "md" | "lg";
}

const variants: Record<string, string> = {
  default: "bg-meruem-accent text-slate-900 hover:bg-sky-400",
  outline: "border border-meruem-border text-meruem-text hover:bg-meruem-panel",
  ghost: "text-meruem-muted hover:text-meruem-text hover:bg-meruem-panel",
  danger: "bg-meruem-red/20 text-meruem-red border border-meruem-red/40 hover:bg-meruem-red/30",
  success: "bg-meruem-green/20 text-meruem-green border border-meruem-green/40 hover:bg-meruem-green/30",
};

const sizes: Record<string, string> = {
  sm: "px-2 py-1 text-xs",
  md: "px-3 py-1.5 text-sm",
  lg: "px-4 py-2 text-base",
};

export function Button({ variant = "default", size = "md", className, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "rounded-md font-medium transition-colors disabled:opacity-40 disabled:pointer-events-none",
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  );
}