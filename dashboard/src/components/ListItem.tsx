"use client";

import { cn } from "@/lib/utils";
import { IconChevronRight } from "@tabler/icons-react";

interface ListItemProps {
  title: string;
  description?: string;
  href?: string;
  onClick?: () => void;
  action?: React.ReactNode;
  icon?: React.ReactNode;
  className?: string;
}

export default function ListItem({
  title,
  description,
  href,
  onClick,
  action,
  icon,
  className,
}: ListItemProps) {
  const Tag = href ? "a" : "button";
  const tagProps = href ? { href } : { onClick, type: "button" as const };

  return (
    <Tag
      {...tagProps}
      className={cn(
        "group w-full flex items-center gap-3 px-3 py-2.5 rounded-lg border border-white/[0.06] hover:bg-white/[0.03] transition-colors text-left",
        className
      )}
    >
      {icon && (
        <div className="shrink-0 text-muted-foreground/60">{icon}</div>
      )}
      <div className="flex-1 min-w-0">
        <span className="text-[13px] font-medium text-foreground">{title}</span>
        {description && (
          <p className="text-[11px] text-muted-foreground/60 mt-0.5 line-clamp-1">
            {description}
          </p>
        )}
      </div>
      {action}
      {href && !action && (
        <IconChevronRight className="size-3.5 text-muted-foreground/20 group-hover:text-muted-foreground/40 transition-colors shrink-0" />
      )}
    </Tag>
  );
}
