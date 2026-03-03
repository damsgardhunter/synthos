import { Link, useLocation } from "wouter";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarFooter,
} from "@/components/ui/sidebar";
import {
  LayoutDashboard,
  Atom,
  Database,
  FlaskConical,
  Cpu,
  FileText,
  Zap,
  Magnet,
  Activity,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import type { LearningPhase } from "@shared/schema";
import { Progress } from "@/components/ui/progress";

const navItems = [
  { title: "Command Center", url: "/", icon: LayoutDashboard },
  { title: "Atomic Explorer", url: "/atoms", icon: Atom },
  { title: "Materials Database", url: "/materials", icon: Database },
  { title: "Novel Discovery", url: "/discovery", icon: FlaskConical },
  { title: "Superconductor Lab", url: "/superconductor", icon: Magnet },
  { title: "Computational Physics", url: "/physics", icon: Activity },
  { title: "Research Pipeline", url: "/research", icon: FileText },
];

export function AppSidebar() {
  const [location] = useLocation();
  const { data: phases } = useQuery<LearningPhase[]>({ queryKey: ["/api/learning-phases"] });

  const activePhase = phases?.find(p => p.status === "active");
  const overallProgress = phases ? phases.reduce((sum, p) => sum + p.progress, 0) / Math.max(phases.length, 1) : 0;

  return (
    <Sidebar>
      <SidebarHeader className="border-b border-sidebar-border px-4 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary">
            <Cpu className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <p className="text-sm font-semibold leading-none text-sidebar-foreground">MatSci-∞</p>
            <p className="text-xs text-muted-foreground mt-0.5">Supercomputer v1.0</p>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navItems.map((item) => {
                const isActive = location === item.url;
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild data-active={isActive}>
                      <Link href={item.url} data-testid={`nav-${item.title.toLowerCase().replace(/ /g, "-")}`}>
                        <item.icon className="h-4 w-4" />
                        <span>{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>System Status</SidebarGroupLabel>
          <SidebarGroupContent className="px-2">
            <div className="space-y-3 px-2 py-1">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-muted-foreground">Overall Learning</span>
                  <span className="text-xs font-mono font-medium text-primary">{overallProgress.toFixed(1)}%</span>
                </div>
                <Progress value={overallProgress} className="h-1.5" />
              </div>
              {activePhase && (
                <div>
                  <div className="flex items-center gap-1.5 mb-1">
                    <div className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                    <span className="text-xs text-muted-foreground truncate">Active: {activePhase.name.split(" ").slice(0, 2).join(" ")}</span>
                  </div>
                  <Progress value={activePhase.progress} className="h-1.5" />
                  <p className="text-xs font-mono text-primary mt-1">{activePhase.progress.toFixed(1)}%</p>
                </div>
              )}
            </div>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-sidebar-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Zap className="h-3.5 w-3.5 text-primary" />
          <span className="text-xs text-muted-foreground">NIST · MP · OQMD · AFLOW</span>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
