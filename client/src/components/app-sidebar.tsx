import { Link, useLocation } from "wouter";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarFooter,
} from "@/components/ui/sidebar";
import {
  LayoutDashboard,
  BarChart3,
  Atom,
  FlaskConical,
  Zap,
  ChevronDown,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import type { LearningPhase } from "@shared/schema";
import { Progress } from "@/components/ui/progress";
import { useWebSocket, type EngineTempo } from "@/hooks/use-websocket";
import { useState } from "react";

interface NavItem {
  title: string;
  url: string;
  icon: any;
  children?: { title: string; url: string }[];
}

const navItems: NavItem[] = [
  {
    title: "Command Center",
    url: "/",
    icon: LayoutDashboard,
  },
  {
    title: "Pipeline Statistics",
    url: "/research",
    icon: BarChart3,
    children: [
      { title: "Learning Phases", url: "/research" },
      { title: "Multi-Fidelity Pipeline", url: "/research?tab=multi-fidelity" },
      { title: "Synthesis Variables", url: "/research?tab=synthesis-vars" },
      { title: "Advanced Physics", url: "/research?tab=advanced-physics" },
    ],
  },
  {
    title: "Materials Explorer",
    url: "/materials",
    icon: Atom,
  },
  {
    title: "Discovery Lab",
    url: "/discovery-lab",
    icon: FlaskConical,
    children: [
      { title: "Pipeline & DFT", url: "/discovery-lab" },
      { title: "Causal Physics", url: "/causal-physics" },
      { title: "Inverse Design", url: "/discovery-lab?tab=next-gen-pipeline" },
      { title: "Theory Discovery", url: "/discovery-lab?tab=theory-discovery" },
      { title: "Topo Invariants", url: "/discovery-lab?tab=topo-invariants" },
      { title: "Doping Engine", url: "/discovery-lab?tab=doping-engine" },
    ],
  },
];

const TEMPO_STYLES: Record<string, { dot: string; label: string }> = {
  excited: { dot: "bg-green-500", label: "Excited" },
  exploring: { dot: "bg-[hsl(var(--gold))]", label: "Exploring" },
  contemplating: { dot: "bg-amber-500", label: "Contemplating" },
};

function SynthosLogo({ size = 48 }: { size?: number }) {
  // Uses the actual logo PNG if available, falls back to SVG
  return (
    <div className="flex-shrink-0" style={{ width: size, height: size }}>
      <img
        src="/synthos-logo.png"
        alt="Synthos"
        width={size}
        height={size}
        className="w-full h-full object-contain"
        onError={(e) => {
          // Hide broken image, show SVG fallback
          (e.target as HTMLImageElement).style.display = "none";
          const fallback = (e.target as HTMLImageElement).nextElementSibling;
          if (fallback) (fallback as HTMLElement).style.display = "block";
        }}
      />
      <svg width={size} height={size} viewBox="0 0 120 120" style={{ display: "none" }}>
        {/* Geometric gold trunk */}
        <polygon points="55,90 58,55 62,55 65,90" fill="#B8A04A" />
        <polygon points="58,55 60,50 62,55" fill="#C5A55A" />
        <polygon points="55,90 50,98 58,95" fill="#8B7340" />
        <polygon points="65,90 70,98 62,95" fill="#8B7340" />
        {/* Network branches — steel blue */}
        <line x1="60" y1="50" x2="40" y2="35" stroke="#8FAEC0" strokeWidth="2.5" />
        <line x1="60" y1="50" x2="80" y2="35" stroke="#8FAEC0" strokeWidth="2.5" />
        <line x1="60" y1="50" x2="60" y2="18" stroke="#8FAEC0" strokeWidth="2.5" />
        <line x1="60" y1="50" x2="30" y2="50" stroke="#8FAEC0" strokeWidth="2" />
        <line x1="60" y1="50" x2="90" y2="50" stroke="#8FAEC0" strokeWidth="2" />
        <line x1="40" y1="35" x2="25" y2="25" stroke="#8FAEC0" strokeWidth="1.8" />
        <line x1="80" y1="35" x2="95" y2="25" stroke="#8FAEC0" strokeWidth="1.8" />
        <line x1="40" y1="35" x2="30" y2="50" stroke="#8FAEC0" strokeWidth="1.5" />
        <line x1="80" y1="35" x2="90" y2="50" stroke="#8FAEC0" strokeWidth="1.5" />
        <line x1="60" y1="18" x2="40" y2="15" stroke="#8FAEC0" strokeWidth="1.5" />
        <line x1="60" y1="18" x2="80" y2="15" stroke="#8FAEC0" strokeWidth="1.5" />
        <line x1="40" y1="15" x2="25" y2="25" stroke="#8FAEC0" strokeWidth="1.5" />
        <line x1="80" y1="15" x2="95" y2="25" stroke="#8FAEC0" strokeWidth="1.5" />
        <line x1="30" y1="50" x2="25" y2="60" stroke="#8FAEC0" strokeWidth="1.5" />
        <line x1="90" y1="50" x2="95" y2="60" stroke="#8FAEC0" strokeWidth="1.5" />
        {/* Blue junction nodes */}
        <circle cx="60" cy="50" r="4" fill="#8FAEC0" />
        <circle cx="40" cy="35" r="3.5" fill="#8FAEC0" />
        <circle cx="80" cy="35" r="3.5" fill="#8FAEC0" />
        <circle cx="60" cy="18" r="3.5" fill="#8FAEC0" />
        <circle cx="40" cy="15" r="2.5" fill="#8FAEC0" />
        <circle cx="80" cy="15" r="2.5" fill="#8FAEC0" />
        {/* Dark red gem nodes */}
        <circle cx="25" cy="25" r="5.5" fill="#6B2020" />
        <circle cx="95" cy="25" r="5.5" fill="#6B2020" />
        <circle cx="30" cy="50" r="4.5" fill="#7A2828" />
        <circle cx="90" cy="50" r="4.5" fill="#7A2828" />
        <circle cx="25" cy="60" r="4" fill="#6B2020" />
        <circle cx="95" cy="60" r="4" fill="#6B2020" />
        <circle cx="50" cy="25" r="3.5" fill="#7A2828" />
        <circle cx="70" cy="25" r="3.5" fill="#7A2828" />
        {/* White diamond nodes */}
        <circle cx="45" cy="45" r="2.5" fill="#E0DDD5" />
        <circle cx="75" cy="45" r="2.5" fill="#E0DDD5" />
        <circle cx="55" cy="12" r="2.5" fill="#E0DDD5" />
        <circle cx="65" cy="12" r="2.5" fill="#E0DDD5" />
      </svg>
    </div>
  );
}

export function AppSidebar() {
  const [location] = useLocation();
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const { data: phases } = useQuery<LearningPhase[]>({ queryKey: ["/api/learning-phases"] });
  const ws = useWebSocket();

  const activePhase = phases?.find(p => p.status === "active");
  const overallProgress = phases ? phases.reduce((sum, p) => sum + p.progress, 0) / Math.max(phases.length, 1) : 0;

  const tempoStyle = TEMPO_STYLES[ws.tempo] ?? TEMPO_STYLES.exploring;
  const isRunning = ws.engineState === "running";
  const pulseSpeed = ws.tempo === "excited" ? "animate-pulse" : ws.tempo === "contemplating" ? "animate-[pulse_3s_ease-in-out_infinite]" : "animate-pulse";

  const toggleExpand = (title: string) => {
    setExpandedItems(prev => {
      const next = new Set(prev);
      if (next.has(title)) next.delete(title);
      else next.add(title);
      return next;
    });
  };

  const isActiveRoute = (item: NavItem) => {
    if (item.url === "/" && location === "/") return true;
    if (item.url !== "/" && location.startsWith(item.url.split("?")[0])) return true;
    if (item.children?.some(c => location.startsWith(c.url.split("?")[0]))) return true;
    return false;
  };

  return (
    <Sidebar>
      <SidebarHeader className="border-b border-[hsl(var(--gold)/0.15)] px-5 py-5">
        <div className="flex flex-col items-center gap-3">
          <SynthosLogo />
          <div className="text-center">
            <h1 className="synthos-heading text-xl gold-text tracking-[0.2em]">SYNTHOS</h1>
            <p className="text-[10px] text-[hsl(var(--gold-muted))] mt-1 tracking-widest uppercase">Materials Intelligence</p>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent className="px-2 pt-4">
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {navItems.map((item) => {
                const active = isActiveRoute(item);
                const isExpanded = expandedItems.has(item.title);
                const hasChildren = item.children && item.children.length > 0;

                return (
                  <SidebarMenuItem key={item.title} className="mb-1">
                    <div className="flex items-center">
                      <SidebarMenuButton
                        asChild
                        data-active={active}
                        className={`flex-1 py-3 px-3 rounded-lg transition-all ${
                          active
                            ? "bg-[hsl(var(--gold)/0.1)] text-[hsl(var(--gold-light))] border border-[hsl(var(--gold)/0.25)]"
                            : "text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--gold-light))] hover:bg-[hsl(var(--gold)/0.05)]"
                        }`}
                      >
                        <Link href={item.url} data-testid={`nav-${item.title.toLowerCase().replace(/ /g, "-")}`}>
                          <item.icon className={`h-4 w-4 ${active ? "text-[hsl(var(--gold))]" : ""}`} />
                          <span className="font-semibold text-sm tracking-wide">{item.title}</span>
                        </Link>
                      </SidebarMenuButton>
                      {hasChildren && (
                        <button
                          onClick={() => toggleExpand(item.title)}
                          className="p-1.5 rounded-md hover:bg-[hsl(var(--gold)/0.08)] transition-colors"
                        >
                          <ChevronDown className={`h-3.5 w-3.5 text-[hsl(var(--muted-foreground))] transition-transform ${isExpanded ? "rotate-180" : ""}`} />
                        </button>
                      )}
                    </div>
                    {hasChildren && isExpanded && (
                      <div className="ml-7 mt-1 space-y-0.5 border-l border-[hsl(var(--gold)/0.12)] pl-3">
                        {item.children!.map(child => {
                          const childActive = location === child.url.split("?")[0] || location + window.location.search === child.url;
                          return (
                            <Link
                              key={child.title}
                              href={child.url}
                              className={`block text-xs py-1.5 px-2 rounded transition-colors ${
                                childActive
                                  ? "text-[hsl(var(--gold))] bg-[hsl(var(--gold)/0.08)]"
                                  : "text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--gold-light))]"
                              }`}
                            >
                              {child.title}
                            </Link>
                          );
                        })}
                      </div>
                    )}
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* System Status */}
        <SidebarGroup className="mt-auto">
          <div className="px-3 py-3 rounded-lg border border-[hsl(var(--gold)/0.12)] bg-[hsl(var(--gold)/0.03)]">
            <div className="space-y-3">
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[10px] uppercase tracking-wider text-[hsl(var(--gold-muted))]">Overall Learning</span>
                  <span className="text-[10px] font-mono font-medium text-[hsl(var(--gold))]">{overallProgress.toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-[hsl(var(--gold)/0.1)] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-[hsl(var(--gold-dark))] to-[hsl(var(--gold))] transition-all"
                    style={{ width: `${overallProgress}%` }}
                  />
                </div>
              </div>
              {activePhase && (
                <div>
                  <div className="flex items-center gap-1.5 mb-1">
                    <div className={`h-1.5 w-1.5 rounded-full ${isRunning ? tempoStyle.dot : "bg-gray-500"} ${isRunning ? pulseSpeed : ""}`} />
                    <span className="text-[10px] text-[hsl(var(--muted-foreground))] truncate">
                      Active: {activePhase.name.split(" ").slice(0, 2).join(" ")}
                    </span>
                  </div>
                  <div className="h-1.5 bg-[hsl(var(--gold)/0.1)] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-[hsl(var(--gold)/0.6)] transition-all"
                      style={{ width: `${activePhase.progress}%` }}
                    />
                  </div>
                </div>
              )}
              {isRunning && (
                <div className="space-y-1" data-testid="engine-status-message">
                  <div className="flex items-center gap-1.5">
                    <div className={`h-1.5 w-1.5 rounded-full ${tempoStyle.dot} ${pulseSpeed}`} />
                    <span className="text-[10px] font-medium text-[hsl(var(--muted-foreground))]">{tempoStyle.label}</span>
                  </div>
                  {ws.statusMessage && (
                    <p className="text-[10px] text-[hsl(var(--muted-foreground)/0.7)] leading-relaxed pl-3">
                      {ws.statusMessage}
                    </p>
                  )}
                </div>
              )}
              {ws.engineState === "paused" && (
                <div className="flex items-center gap-1.5">
                  <div className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                  <span className="text-[10px] text-[hsl(var(--muted-foreground))]">Research paused</span>
                </div>
              )}
              {ws.engineState === "stopped" && (
                <div className="flex items-center gap-1.5">
                  <div className="h-1.5 w-1.5 rounded-full bg-gray-500" />
                  <span className="text-[10px] text-[hsl(var(--muted-foreground))]">Engine offline</span>
                </div>
              )}
            </div>
          </div>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-[hsl(var(--gold)/0.12)] px-5 py-3">
        <div className="flex items-center justify-center gap-2">
          <Zap className="h-3 w-3 text-[hsl(var(--gold-muted))]" />
          <span className="text-[10px] text-[hsl(var(--gold-muted))] tracking-wider">NIST · MP · OQMD · AFLOW</span>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
