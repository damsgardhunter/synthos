import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/not-found";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { ErrorBoundary } from "@/components/error-boundary";
import Dashboard from "@/pages/dashboard";
import MaterialsExplorer from "@/pages/materials-explorer";
import ResearchPipeline from "@/pages/research-pipeline";
import DiscoveryLab from "@/pages/discovery-lab";
import CausalPhysics from "@/pages/causal-physics";
import CandidateDetail from "@/pages/candidate-detail";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/materials" component={MaterialsExplorer} />
      <Route path="/research" component={ResearchPipeline} />
      <Route path="/discovery-lab" component={DiscoveryLab} />
      <Route path="/causal-physics" component={CausalPhysics} />
      <Route path="/candidate/:formula" component={CandidateDetail} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  const sidebarStyle = {
    "--sidebar-width": "17rem",
    "--sidebar-width-icon": "3.5rem",
  };

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <SidebarProvider style={sidebarStyle as React.CSSProperties}>
          <div className="flex h-screen w-full overflow-hidden">
            <AppSidebar />
            <div className="flex flex-col flex-1 min-w-0">
              <header className="flex items-center gap-3 px-5 py-2.5 border-b border-[hsl(var(--gold)/0.12)] bg-background/90 backdrop-blur sticky top-0 z-50">
                <SidebarTrigger data-testid="button-sidebar-toggle" className="text-[hsl(var(--gold-muted))] hover:text-[hsl(var(--gold))]" />
                <div className="h-4 w-px bg-[hsl(var(--gold)/0.15)]" />
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                  <span className="text-xs text-[hsl(var(--gold-muted))] tracking-wide">Online</span>
                </div>
                <div className="ml-auto flex items-center gap-3">
                  <span className="text-xs text-[hsl(var(--gold-muted))] hidden sm:block tracking-wider">
                    Additional Info About MatSci
                  </span>
                </div>
              </header>
              <main className="flex-1 overflow-auto">
                <ErrorBoundary pageName="App">
                  <Router />
                </ErrorBoundary>
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
