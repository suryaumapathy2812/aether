"use client";

/**
 * Minimal input — bottom border, floating label above.
 * Light Phone aesthetic: sparse, quiet, functional.
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
      <label className="block text-[10px] tracking-[0.15em] uppercase text-[var(--color-text-muted)] mb-2 font-normal">
        {label}
      </label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder || label.toLowerCase()}
        className="w-full"
      />
    </div>
  );
}
