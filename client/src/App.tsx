import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/not-found";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import Dashboard from "@/pages/dashboard";
import AtomicExplorer from "@/pages/atomic-explorer";
import MaterialsDatabase from "@/pages/materials-database";
import NovelDiscovery from "@/pages/novel-discovery";
import ResearchPipeline from "@/pages/research-pipeline";
import SuperconductorLab from "@/pages/superconductor-lab";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/atoms" component={AtomicExplorer} />
      <Route path="/materials" component={MaterialsDatabase} />
      <Route path="/discovery" component={NovelDiscovery} />
      <Route path="/superconductor" component={SuperconductorLab} />
      <Route path="/research" component={ResearchPipeline} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  const sidebarStyle = {
    "--sidebar-width": "18rem",
    "--sidebar-width-icon": "3.5rem",
  };

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <SidebarProvider style={sidebarStyle as React.CSSProperties}>
          <div className="flex h-screen w-full overflow-hidden">
            <AppSidebar />
            <div className="flex flex-col flex-1 min-w-0">
              <header className="flex items-center gap-3 px-4 py-2 border-b border-border bg-background/80 backdrop-blur sticky top-0 z-50">
                <SidebarTrigger data-testid="button-sidebar-toggle" />
                <div className="h-4 w-px bg-border" />
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                  <span className="text-xs font-mono text-muted-foreground">System Online</span>
                </div>
                <div className="ml-auto flex items-center gap-3">
                  <span className="text-xs font-mono text-muted-foreground hidden sm:block">
                    MatSci-∞ Supercomputer · Materials Science AI
                  </span>
                </div>
              </header>
              <main className="flex-1 overflow-auto">
                <Router />
              </main>
            </div>
          </div>
        </SidebarProvider>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
