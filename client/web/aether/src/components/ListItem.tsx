import { cn } from "#/lib/utils";
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
  const wrapperClass = cn(
    "group w-full flex items-center gap-3 px-3 py-2.5 rounded-lg shadow-sm hover:bg-muted transition-colors text-left bg-muted/50 mb-3",
    className,
  );

  // When there's an action button, use a div wrapper to avoid nested buttons
  if (action) {
    return (
      <div className={wrapperClass}>
        {icon && (
          <div className="shrink-0 text-muted-foreground/60">{icon}</div>
        )}
        <div className="flex-1 min-w-0">
          <span className="text-sm font-medium text-foreground">{title}</span>
          {description && (
            <p className="text-sm text-muted-foreground/60 mt-0.5 line-clamp-1">
              {description}
            </p>
          )}
        </div>
        {action}
      </div>
    );
  }

  const Tag = href ? "a" : "button";
  const tagProps = href ? { href } : { onClick, type: "button" as const };

  return (
    <Tag {...tagProps} className={wrapperClass}>
      {icon && <div className="shrink-0 text-muted-foreground/60">{icon}</div>}
      <div className="flex-1 min-w-0">
        <span className="text-sm font-medium text-foreground">{title}</span>
        {description && (
          <p className="text-sm text-muted-foreground/60 mt-0.5 line-clamp-1">
            {description}
          </p>
        )}
      </div>
      {href && (
        <IconChevronRight className="size-3.5 text-muted-foreground/20 group-hover:text-muted-foreground/40 transition-colors shrink-0" />
      )}
    </Tag>
  );
}
