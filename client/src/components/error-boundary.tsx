import React from "react";

interface Props {
  children: React.ReactNode;
  /** Label shown in the error UI and sent with the report (e.g. "Dashboard") */
  pageName?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Catches render-time crashes in any child tree and:
 *  1. Reports them to /api/client-errors so the QAE monitor can detect them.
 *  2. Shows a minimal recovery UI instead of a blank white screen.
 *
 * Wrap each page (or the whole <Router />) with this component.
 */
export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Fire-and-forget — never let reporting itself crash the app
    try {
      fetch("/api/client-errors", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "render-crash",
          page: window.location.pathname,
          message: `[${this.props.pageName ?? "unknown page"}] ${error.message}`,
          stack: (error.stack ?? "").slice(0, 2000),
        }),
      }).catch(() => {});
    } catch {
      // ignore
    }
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-full gap-4 p-8 text-center">
          <div className="text-destructive text-4xl">⚠</div>
          <h2 className="text-lg font-semibold">
            {this.props.pageName ? `${this.props.pageName} crashed` : "Page crashed"}
          </h2>
          <p className="text-sm text-muted-foreground max-w-md">
            {this.state.error?.message ?? "An unexpected error occurred."}
          </p>
          <button
            onClick={this.handleRetry}
            className="px-4 py-2 text-sm rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
