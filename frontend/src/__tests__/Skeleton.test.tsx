import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
  Skeleton,
  SkeletonCard,
  SkeletonTable,
  SkeletonChart,
  SkeletonGrid,
} from '../components/ui/Skeleton';

describe('Skeleton', () => {
  it('renders with default props', () => {
    render(<Skeleton />);
    const skeleton = screen.getByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeleton).toBeInTheDocument();
  });

  it('renders text variant', () => {
    render(<Skeleton variant="text" />);
    const skeleton = screen.getByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeleton).toHaveClass('rounded');
  });

  it('renders circular variant', () => {
    render(<Skeleton variant="circular" width={50} height={50} />);
    const skeleton = screen.getByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeleton).toHaveClass('rounded-full');
  });

  it('renders rectangular variant', () => {
    render(<Skeleton variant="rectangular" width={100} height={100} />);
    const skeleton = screen.getByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeleton).toHaveClass('rounded-lg');
  });

  it('applies custom width and height', () => {
    render(<Skeleton width={200} height={50} />);
    const skeleton = screen.getByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeleton).toHaveStyle({ width: '200px', height: '50px' });
  });

  it('applies pulse animation by default', () => {
    render(<Skeleton />);
    const skeleton = screen.getByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeleton).toHaveClass('animate-pulse');
  });

  it('applies wave animation when specified', () => {
    render(<Skeleton animation="wave" />);
    const skeleton = screen.getByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeleton).toHaveClass('animate-shimmer');
  });

  it('applies no animation when specified', () => {
    render(<Skeleton animation="none" />);
    const skeleton = screen.getByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeleton).not.toHaveClass('animate-pulse', 'animate-shimmer');
  });

  it('has aria-hidden attribute', () => {
    render(<Skeleton />);
    const skeleton = screen.getByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeleton).toHaveAttribute('aria-hidden', 'true');
  });
});

describe('SkeletonCard', () => {
  it('renders with default lines', () => {
    render(<SkeletonCard />);
    const skeletons = screen.getAllByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('renders with custom line count', () => {
    render(<SkeletonCard lines={5} />);
    const skeletons = screen.getAllByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeletons.length).toBe(6);
  });
});

describe('SkeletonTable', () => {
  it('renders with default rows and columns', () => {
    render(<SkeletonTable />);
    const skeletons = screen.getAllByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeletons.length).toBe(24);
  });

  it('renders with custom rows and columns', () => {
    render(<SkeletonTable rows={3} columns={3} />);
    const skeletons = screen.getAllByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeletons.length).toBe(12);
  });
});

describe('SkeletonChart', () => {
  it('renders', () => {
    render(<SkeletonChart />);
    const skeletons = screen.getAllByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeletons.length).toBe(2);
  });
});

describe('SkeletonGrid', () => {
  it('renders with default props', () => {
    render(<SkeletonGrid />);
    const skeletons = screen.getAllByText((content, element) => {
      return element?.getAttribute('aria-hidden') === 'true';
    });
    expect(skeletons.length).toBeGreaterThan(0);
  });
});
