"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";

/**
 * Menu list — stacked bordered items, Light Phone style.
 * Items share borders (no double-border between items).
 * Built on shadcn Button (ghost variant) as link.
 */
export default function MenuList({
  items,
}: {
  items: { label: string; href: string }[];
}) {
  return (
    <nav className="w-full max-w-xs">
      {items.map((item) => (
        <Button
          key={item.href}
          variant="aether-menu"
          size="aether-menu"
          asChild
        >
          <Link href={item.href}>{item.label}</Link>
        </Button>
      ))}
    </nav>
  );
}
