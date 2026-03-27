import { Suspense, useEffect, useState } from "react";
import { useNavigate, useLocation } from "@tanstack/react-router";
import { useTheme } from "#/components/ThemeProvider";
import { cn } from "#/lib/utils";
import { useSession } from "#/lib/auth-client";
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
} from "@tabler/icons-react";

const NAV_ITEMS = [
  { href: "/devices", label: "Devices" },
  { href: "/memory", label: "Memory" },
  { href: "/integrations", label: "Integrations" },
  { href: "/skills", label: "Skills" },
] as const;

const HIDE_TOOLBAR = ["/"];
const COMMAND_HINTS = [
  { keys: ["Cmd", "K"], label: "Command" },
  { keys: ["Cmd", "B"], label: "Sessions" },
  { keys: ["Cmd", "N"], label: "New chat" },
  { keys: ["G", "C"], label: "Chat" },
  { keys: ["G", "D"], label: "Devices" },
  { keys: ["G", "M"], label: "Memory" },
  { keys: ["G", "P"], label: "Integrations" },
  { keys: ["G", "S"], label: "Settings" },
] as const;

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
  const { resolvedTheme, setTheme } = useTheme();
  const [showCommandHints, setShowCommandHints] = useState(false);

  useEffect(() => {
    const isMac = /Mac|iPhone|iPad|iPod/i.test(navigator.platform);
    if (!isMac) return;

    const syncHints = (event: KeyboardEvent) => {
      if (event.metaKey || event.key === "Meta") {
        setShowCommandHints(true);
        return;
      }
      setShowCommandHints(false);
    };

    const hideHints = () => {
      setShowCommandHints(false);
    };

    window.addEventListener("keydown", syncHints);
    window.addEventListener("keyup", syncHints);
    window.addEventListener("blur", hideHints);
    document.addEventListener("visibilitychange", hideHints);

    return () => {
      window.removeEventListener("keydown", syncHints);
      window.removeEventListener("keyup", syncHints);
      window.removeEventListener("blur", hideHints);
      document.removeEventListener("visibilitychange", hideHints);
    };
  }, []);

  if (isPending || !session || pathname === "/login" || HIDE_TOOLBAR.includes(pathname)) {
    return null;
  }

  const toggleTheme = () =>
    setTheme(resolvedTheme === "dark" ? "light" : "dark");

  return (
    <>
      {/* Gradient fade from background to transparent */}
      <div className="fixed top-0 left-0 right-0 z-30 h-20 bg-gradient-to-b from-background to-transparent pointer-events-none" />

      <div
        className={cn(
          "fixed top-3 left-1/2 z-35 hidden -translate-x-1/2 items-center gap-2 rounded-full border border-border/70 bg-background/85 px-3 py-2 shadow-lg backdrop-blur-md md:flex",
          showCommandHints
            ? "pointer-events-none opacity-100 translate-y-0"
            : "pointer-events-none opacity-0 -translate-y-2",
          "transition-all duration-150",
        )}
        aria-hidden={!showCommandHints}
      >
        {COMMAND_HINTS.map((hint) => (
          <div
            key={`${hint.label}-${hint.keys.join("-")}`}
            className="flex items-center gap-2 rounded-full bg-muted/60 px-2 py-1"
          >
            <div className="flex items-center gap-1">
              {hint.keys.map((key) => (
                <kbd
                  key={key}
                  className="inline-flex h-5 min-w-5 items-center justify-center rounded border border-border/70 bg-background px-1.5 text-[10px] font-medium text-muted-foreground shadow-sm"
                >
                  {key}
                </kbd>
              ))}
            </div>
            <span className="text-[11px] font-medium text-foreground/80">
              {hint.label}
            </span>
          </div>
        ))}
      </div>

      <div className="fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-4 py-3 pointer-events-none">
      {/* Left — logo (new chat) */}
      <div className="flex items-center pointer-events-auto">
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

        {NAV_ITEMS.map((item) => {
          const isActive = pathname.startsWith(item.href);
          return (
            <Tooltip key={item.href}>
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
    </>
  );
}
