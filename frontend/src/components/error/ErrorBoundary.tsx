import { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  level?: 'app' | 'page' | 'widget';
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });

    console.error('ErrorBoundary caught:', {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      url: window.location.href,
    });

    this.reportToBackend(error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  private reportToBackend = async (error: Error, errorInfo: ErrorInfo): Promise<void> => {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) return;

      await fetch('/api/system/error', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          error: {
            message: error.message,
            name: error.name,
            stack: error.stack,
          },
          errorInfo: {
            componentStack: errorInfo.componentStack,
          },
          context: {
            timestamp: new Date().toISOString(),
            url: window.location.href,
            userAgent: navigator.userAgent,
          },
        }),
      });
    } catch (reportingError) {
      console.error('Failed to report error to backend:', reportingError);
    }
  };

  private handleRetry = (): void => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  private handleReload = (): void => {
    window.location.reload();
  };

  private handleGoHome = (): void => {
    window.location.href = '/dashboard';
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const isWidget = this.props.level === 'widget';
      const isApp = this.props.level === 'app';

      return (
        <div
          className={`
          flex flex-col items-center justify-center p-6
          bg-gradient-to-br from-red-500/10 to-red-500/5
          border border-red-500/20 rounded-xl
          ${isApp ? 'min-h-screen' : isWidget ? 'min-h-[200px]' : 'min-h-[400px]'}
        `}
        >
          <div className="w-16 h-16 mb-4 rounded-full bg-red-500/20 flex items-center justify-center">
            <AlertTriangle className="w-8 h-8 text-red-400" />
          </div>

          <h2 className="text-xl font-bold text-white mb-2">
            {isWidget
              ? 'Component Error'
              : isApp
                ? 'Something went wrong'
                : 'Page could not be loaded'}
          </h2>

          <p className="text-gray-400 text-center mb-6 max-w-md">
            {this.state.error?.message || 'An unexpected error occurred.'}
            {!isWidget && ' Please try again.'}
          </p>

          <div className="flex gap-3">
            {!isWidget && (
              <button
                onClick={this.handleGoHome}
                className="px-6 py-2 bg-gray-700 hover:bg-gray-600 
                           text-white rounded-lg transition-colors
                           flex items-center gap-2"
              >
                <Home className="w-4 h-4" />
                Go to Dashboard
              </button>
            )}

            <button
              onClick={this.handleRetry}
              className="px-6 py-2 bg-cyan-500 hover:bg-cyan-400 
                         text-white rounded-lg transition-colors
                         flex items-center gap-2 shadow-lg shadow-cyan-500/20"
            >
              <RefreshCw className="w-4 h-4" />
              Try Again
            </button>
          </div>

          {process.env.NODE_ENV === 'development' && this.state.error?.stack && (
            <details className="mt-6 w-full max-w-2xl">
              <summary className="text-gray-500 cursor-pointer hover:text-gray-400">
                Debug Information (Developers Only)
              </summary>
              <pre className="mt-2 p-4 bg-black/50 rounded text-xs text-red-300 overflow-auto max-h-64">
                {this.state.error.stack}
              </pre>
            </details>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
