import { Input } from "#/components/ui/input";
import { Label } from "#/components/ui/label";

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
      <Label className="block text-xs tracking-[0.15em] uppercase text-muted-foreground mb-2 font-normal">
        {label}
      </Label>
      <Input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder || label.toLowerCase()}
        className="bg-accent/50 border border-border rounded-full shadow-none px-4 py-2.5 text-base font-medium tracking-[0.01em] focus-visible:ring-0 focus-visible:border-ring focus-visible:bg-accent/80 transition-colors duration-300 h-auto"
      />
    </div>
  );
}
