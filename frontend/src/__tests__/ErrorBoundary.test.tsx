import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ErrorBoundary } from '../components/error/ErrorBoundary';

// Helper component that throws error
function BrokenComponent() {
  throw new Error('Test error');
}

describe('ErrorBoundary', () => {
  // Suppress console.error for cleaner test output
  const originalConsoleError = console.error;
  beforeEach(() => {
    console.error = vi.fn();
  });

  afterEach(() => {
    console.error = originalConsoleError;
  });

  it('renders children when no error', () => {
    render(
      <ErrorBoundary>
        <div data-testid="child">Working</div>
      </ErrorBoundary>
    );

    expect(screen.getByTestId('child')).toBeInTheDocument();
  });

  it('shows fallback when error occurs', () => {
    render(
      <ErrorBoundary>
        <BrokenComponent />
      </ErrorBoundary>
    );

    // Check for error title
    expect(screen.getByText('Page could not be loaded')).toBeInTheDocument();
    // Check for the error message - use regex to match partial text
    expect(screen.getByText(/Test error/)).toBeInTheDocument();
  });

  it('shows custom fallback when provided', () => {
    const customFallback = <div data-testid="custom">Custom Error</div>;

    render(
      <ErrorBoundary fallback={customFallback}>
        <BrokenComponent />
      </ErrorBoundary>
    );

    expect(screen.getByTestId('custom')).toBeInTheDocument();
  });

  it('has retry and home buttons', () => {
    render(
      <ErrorBoundary>
        <BrokenComponent />
      </ErrorBoundary>
    );

    // Use getByRole for buttons
    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /go to dashboard/i })).toBeInTheDocument();
  });
});
