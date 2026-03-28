import { Suspense } from "react";
import { useNavigate, useLocation } from "@tanstack/react-router";
import { useTheme } from "#/components/ThemeProvider";
import { cn } from "#/lib/utils";
import { useSession } from "#/lib/auth-client";
import { useShortcutsContext } from "#/components/KeyboardShortcutsProvider";
import { getCommandPaletteShortcutKeys } from "#/lib/shortcuts";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "#/components/ui/tooltip";
import { Avatar, AvatarFallback, AvatarImage } from "#/components/ui/avatar";
import {
  IconMessage,
  IconMessage2Filled,
  IconBrain,
  IconDeviceMobile,
  IconDeviceMobileFilled,
  IconPlugConnected,
  IconSparkles,
  IconSun,
  IconMoon,
  IconSearch,
} from "@tabler/icons-react";

const NAV_ITEMS = [
  { href: "/devices", label: "Devices" },
  { href: "/memory", label: "Memory" },
  { href: "/integrations", label: "Integrations" },
  { href: "/skills", label: "Skills" },
] as const;

const HIDE_TOOLBAR = ["/"];

function NavIcon({ href, active }: { href: string; active: boolean }) {
  switch (href) {
    case "/devices":
      return active ? <IconDeviceMobileFilled className="size-5" /> : <IconDeviceMobile className="size-5" strokeWidth={1.5} />;
    case "/memory":
      return <IconBrain className="size-5" strokeWidth={active ? 2 : 1.5} />;
    case "/integrations":
      return <IconPlugConnected className="size-5" strokeWidth={active ? 2 : 1.5} />;
    case "/skills":
      return <IconSparkles className="size-5" strokeWidth={active ? 2 : 1.5} />;
    default:
      return null;
  }
}

export default function FloatingToolbar() {
  return (
    <Suspense>
      <FloatingToolbarInner />
    </Suspense>
  );
}

function FloatingToolbarInner() {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const { data: session, isPending } = useSession();
  const { openCommandPalette } = useShortcutsContext();
  const { resolvedTheme, setTheme } = useTheme();

  if (isPending || !session || pathname === "/login" || HIDE_TOOLBAR.includes(pathname)) {
    return null;
  }

  const toggleTheme = () =>
    setTheme(resolvedTheme === "dark" ? "light" : "dark");

  return (
    <>
      {/* Gradient fade from background to transparent */}
      <div className="fixed top-0 left-0 right-0 z-30 h-20 bg-gradient-to-b from-background to-transparent pointer-events-none" />

      <div className="fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-4 py-3 pointer-events-none">
      {/* Left — logo (new chat) */}
      <div className="relative flex items-center pointer-events-auto">
        <button
          onClick={() => navigate({ to: "/chat" })}
          className="flex items-center justify-center rounded-lg transition-colors cursor-pointer"
          aria-label="Home"
        >
          <img
            src={
              resolvedTheme === "dark"
                ? "/icon-animated-white.svg"
                : "/icon-animated-black.svg"
            }
            alt="Aether"
            className="w-8 h-8"
          />
        </button>
      </div>

      {/* Right — nav icons + account */}
      <div className="flex items-center gap-1 pointer-events-auto">
        <div className="relative mr-2">
          <button
            onClick={openCommandPalette}
            className="flex h-9 min-w-28 items-center justify-between rounded-lg border border-border/70 bg-background/80 px-3 text-sm text-muted-foreground shadow-sm backdrop-blur-sm transition-colors hover:bg-accent/60"
            aria-label="Open command menu"
          >
            <IconSearch className="size-5"/>
            <span className="ml-3 inline-flex items-center gap-1">
              {getCommandPaletteShortcutKeys().map((key) => (
                <kbd
                  key={key}
                  className="inline-flex h-5 min-w-5 items-center justify-center rounded border border-border/70 bg-muted/70 px-1 text-[10px] font-medium text-foreground/80"
                >
                  {key}
                </kbd>
              ))}
            </span>
          </button>
        </div>

        <div className="relative">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => navigate({ to: "/sessions" })}
                className={cn(
                  "w-9 h-9 flex items-center justify-center rounded-lg text-foreground transition-colors",
                  pathname.startsWith("/sessions")
                    ? "bg-foreground text-background"
                    : "hover:bg-accent/60"
                )}
                aria-label="Sessions"
              >
                {pathname.startsWith("/sessions") ? (
                  <IconMessage2Filled className="size-5" />
                ) : (
                  <IconMessage className="size-5" strokeWidth={1.5} />
                )}
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom">Sessions</TooltipContent>
          </Tooltip>
        </div>

        {NAV_ITEMS.map((item) => {
          const isActive = pathname.startsWith(item.href);
          return (
            <div key={item.href} className="relative">
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    onClick={() => navigate({ to: item.href })}
                    className={cn(
                      "w-9 h-9 flex items-center justify-center rounded-lg text-foreground transition-colors",
                      isActive
                        ? "bg-foreground text-background"
                        : "hover:bg-accent/60"
                    )}
                    aria-label={item.label}
                  >
                    <NavIcon href={item.href} active={isActive} />
                  </button>
                </TooltipTrigger>
                <TooltipContent side="bottom">{item.label}</TooltipContent>
              </Tooltip>
            </div>
          );
        })}

        {/* Theme toggle */}
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={toggleTheme}
              className="w-9 h-9 flex items-center justify-center rounded-lg text-foreground transition-colors hover:bg-accent/60"
              aria-label="Toggle theme"
            >
              {resolvedTheme === "dark" ? (
                <IconSun className="size-5" strokeWidth={1.5} />
              ) : (
                <IconMoon className="size-5" strokeWidth={1.5} />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Toggle theme</TooltipContent>
        </Tooltip>

        <div className="relative">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => navigate({ to: "/account" })}
                className={cn(
                  "w-9 h-9 flex items-center justify-center rounded-lg transition-colors",
                  pathname.startsWith("/account")
                    ? "bg-foreground text-background"
                    : "text-foreground hover:bg-accent/60"
                )}
                aria-label="Account"
              >
                <Avatar size="sm">
                  {session?.user?.image && (
                    <AvatarImage
                      src={session.user.image}
                      alt={session?.user?.name || ""}
                    />
                  )}
                  <AvatarFallback className="text-sm">
                    {(session?.user?.name || session?.user?.email || "U")
                      .charAt(0)
                      .toUpperCase()}
                  </AvatarFallback>
                </Avatar>
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom">Account</TooltipContent>
          </Tooltip>
        </div>
      </div>
    </div>
    </>
  );
}
