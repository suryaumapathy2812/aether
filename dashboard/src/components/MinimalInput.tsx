"use client";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

/**
 * Minimal input — bottom border only, floating label above.
 * Light Phone aesthetic: sparse, quiet, functional.
 * Built on shadcn Input + Label primitives.
 */
export default function MinimalInput({
  label,
  type = "text",
  value,
  onChange,
  placeholder,
}: {
  label: string;
  type?: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <div className="w-full mb-6">
      <Label className="block text-[10px] tracking-[0.15em] uppercase text-muted-foreground mb-2 font-normal">
        {label}
      </Label>
      <Input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder || label.toLowerCase()}
        className="bg-transparent border-0 border-b border-border rounded-none shadow-none px-0 py-2.5 text-[0.9375rem] font-light tracking-[0.01em] focus-visible:ring-0 focus-visible:border-secondary-foreground transition-colors duration-300 h-auto"
      />
    </div>
  );
}
