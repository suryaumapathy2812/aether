"use client";

import Link from "next/link";

/**
 * Menu list — stacked bordered items, Light Phone style.
 * Items share borders (no double-border between items).
 */
export default function MenuList({
  items,
}: {
  items: { label: string; href: string }[];
}) {
  return (
    <nav className="w-full max-w-xs">
      {items.map((item) => (
        <Link key={item.href} href={item.href} className="menu-item">
          {item.label}
        </Link>
      ))}
    </nav>
  );
}
