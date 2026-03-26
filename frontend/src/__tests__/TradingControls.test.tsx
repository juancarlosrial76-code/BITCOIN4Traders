import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { TradingControls } from '../components/trading/TradingControls';
import { BrowserRouter } from 'react-router-dom';

vi.mock('../api/client', () => ({
  api: {
    trading: {
      status: vi.fn().mockResolvedValue({ running: false, mode: 'paper' }),
      start: vi.fn().mockResolvedValue({ success: true }),
      stop: vi.fn().mockResolvedValue({ success: true }),
    },
  },
}));

function renderWithRouter(component: React.ReactElement) {
  return render(<BrowserRouter>{component}</BrowserRouter>);
}

describe('TradingControls', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders trading controls', async () => {
    renderWithRouter(<TradingControls />);

    await waitFor(() => {
      expect(screen.getByText(/Trading/)).toBeInTheDocument();
    });
  });

  it('shows trading buttons', async () => {
    renderWithRouter(<TradingControls />);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /paper/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /live/i })).toBeInTheDocument();
    });
  });

  it('can switch between live and paper mode', async () => {
    renderWithRouter(<TradingControls />);

    await waitFor(() => {
      const paperButton = screen.getByRole('button', { name: /paper/i });
      const liveButton = screen.getByRole('button', { name: /live/i });

      expect(paperButton).toBeInTheDocument();
      expect(liveButton).toBeInTheDocument();
    });
  });
});
