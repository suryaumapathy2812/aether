"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  MessageCircle, 
  Sparkles, 
  Brain, 
  Zap, 
  User, 
  LayoutGrid,
  MessageSquare
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  motion,
  AnimatePresence,
} from "framer-motion";
import { useUIPreferences } from "@/lib/ui-preferences";

interface DockItemProps {
  href: string;
  label: string;
  icon: React.ElementType;
  isActive?: boolean;
}

function DockItem({ href, label, icon: Icon, isActive }: DockItemProps) {
  const [isHovered, setHovered] = React.useState(false);

  return (
    <div className="relative flex flex-col items-center justify-end w-12 h-12">
      <Link href={href} className="absolute bottom-1 z-10">
        <div
          className={cn(
            "w-10 h-10 rounded-full flex items-center justify-center transition-colors duration-200 cursor-pointer border border-transparent backdrop-blur-md",
            isActive 
              ? "bg-zinc-800 text-foreground shadow-[0_0_20px_rgba(255,255,255,0.1)] border-white/10 z-20" 
              : "text-muted-foreground hover:text-foreground bg-transparent hover:bg-zinc-900 hover:border-white/10"
          )}
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
        >
          <Icon className="w-5 h-5" strokeWidth={1.5} />
          {isActive && (
            <span className="absolute -bottom-1 w-1 h-1 rounded-full bg-foreground shadow-[0_0_8px_rgba(255,255,255,0.5)]" />
          )}
        </div>
      </Link>

      {/* Label appears ABOVE the dock (macOS style) */}
      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="absolute -top-12 left-1/2 -translate-x-1/2 px-4 py-2 rounded-lg bg-zinc-900/90 text-zinc-100 border border-white/10 shadow-xl backdrop-blur-md whitespace-nowrap z-50 pointer-events-none"
          >
            <span className="text-[10px] font-medium tracking-wide">
              {label}
            </span>
            {/* Tiny triangle pointer */}
            <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 border-4 border-transparent border-t-zinc-900/90" />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function Dock() {
  const pathname = usePathname();
  const { dockBehavior } = useUIPreferences();
  const [isHovered, setIsHovered] = React.useState(false);

  // Don't show dock on login page
  if (pathname === "/") return null;

  const items = [
    { href: "/home", label: "Home", icon: LayoutGrid },
    { href: "/chat", label: "Chat", icon: MessageCircle },
    { href: "/agent", label: "Agent", icon: Sparkles },
    { href: "/memory", label: "Memory", icon: Brain },
    { href: "/channels", label: "Channels", icon: MessageSquare },
    { href: "/plugins", label: "Plugins", icon: Zap },
    { href: "/account", label: "Account", icon: User },
  ];

  // Behavior Logic
  // Shrink: Default small scale, scale up on hover. Always visible (y=0).
  // Auto-hide: Default hidden (y=100), visible on hover (y=0).

  const isShrinkMode = dockBehavior === "shrink";
  
  // Animation variants
  const navVariants = {
    visible: { 
      y: 0, 
      opacity: 1,
      scale: 1,
    },
    hidden: { 
      y: 100, 
      opacity: 0,
      scale: 1,
    },
    shrunk: {
      y: 0,
      opacity: 1,
      scale: 0.8, // Slightly smaller when idle
    }
  };

  const currentVariant = isShrinkMode 
    ? (isHovered ? "visible" : "shrunk") 
    : (isHovered ? "visible" : "hidden");

  return (
    <div 
      className="fixed bottom-0 left-0 right-0 h-24 z-50 hidden md:flex flex-col items-center justify-end pb-6 pointer-events-none"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => {
        setIsHovered(false);
      }}
    >
      <motion.nav
        initial="hidden"
        animate={currentVariant}
        variants={navVariants}
        transition={{ type: "spring", stiffness: 200, damping: 20 }}
        style={{ gap: isHovered ? 6 : 8 }}
        className="flex items-end px-4 py-2 rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl shadow-[0_8px_32px_rgba(0,0,0,0.4)] pointer-events-auto origin-bottom overflow-visible"
      >
        {items.map((item) => (
          <DockItem
            key={item.href}
            {...item}
            isActive={pathname.startsWith(item.href)}
          />
        ))}
      </motion.nav>
      
      {/* Hidden Indicator */}
      {!isHovered && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: isShrinkMode ? 0 : 0.5 }}
          className="absolute bottom-1 w-12 h-1 rounded-full bg-white/20 blur-[1px]"
        />
      )}
      
      {/* Trigger zone */}
      <div 
        className="absolute bottom-0 left-0 right-0 h-4 pointer-events-auto" 
        onMouseEnter={() => setIsHovered(true)}
      />
    </div>
  );
}
