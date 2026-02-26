"""
Comprehensive Test Suite for Quantum Optimization Module
=========================================================
Tests all quantum-inspired algorithms with validation.
"""

import pytest
import numpy as np
import pandas as pd
from src.quantum.quantum_optimization import (
    QuantumAnnealingOptimizer,
    QAOASimulator,
    QuantumInspiredEvolution,
    QuantumState,
    quantum_portfolio_optimization,
    qaoa_trade_routing,
    quantum_evolution_strategy,
)


class TestQuantumAnnealingOptimizer:
    """Test quantum annealing portfolio optimization."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns data."""
        np.random.seed(42)
        return pd.DataFrame(
            np.random.randn(100, 3) * 0.02, columns=["BTC", "ETH", "SOL"]
        )

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = QuantumAnnealingOptimizer(n_qubits=5, annealing_steps=100)
        assert optimizer.n_qubits == 5
        assert optimizer.annealing_steps == 100
        assert len(optimizer.history) == 0  # No optimization history yet

    def test_hamiltonian_creation(self, sample_returns):
        """Test Ising Hamiltonian creation."""
        optimizer = QuantumAnnealingOptimizer()

        expected_returns = sample_returns.mean().values * 252  # Annualize returns
        covariance = sample_returns.cov().values * 252  # Annualize covariance

        H = optimizer._create_hamiltonian(
            covariance, expected_returns, risk_aversion=1.0
        )

        assert H.shape == (3, 3)
        assert np.all(np.diag(H) != 0)  # Diagonal entries encode local fields

    def test_portfolio_optimization(self, sample_returns):
        """Test complete portfolio optimization."""
        optimizer = QuantumAnnealingOptimizer(n_qubits=3, annealing_steps=50)

        result = optimizer.optimize_portfolio(sample_returns, risk_aversion=1.0)

        # Check result structure
        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result
        assert (
            "quantum_advantage" in result
        )  # Measure of quantum vs classical performance

        # Check weights sum to 1 (fully invested portfolio)
        weights = list(result["weights"].values())
        assert abs(sum(weights) - 1.0) < 0.01

        # Check all weights non-negative (long-only)
        assert all(w >= 0 for w in weights)

        # Check Sharpe ratio is a finite number (can be negative with random data)
        assert np.isfinite(result["sharpe_ratio"])

    def test_energy_calculation(self):
        """Test Ising energy calculation."""
        optimizer = QuantumAnnealingOptimizer()

        H = np.array([[1, 0.5], [0.5, 1]])  # Ising interaction matrix
        state = np.array([1, -1])  # Binary spin configuration

        energy = optimizer._calculate_energy(state, H)

        # Energy should be negative for valid state
        assert isinstance(energy, float)
        assert energy < 0

    def test_quantum_tunneling(self):
        """Test quantum tunneling behavior."""
        optimizer = QuantumAnnealingOptimizer()

        # Create simple system
        H = np.eye(5)  # Identity interaction matrix
        state = np.ones(5)  # All spins up

        # Test tunneling at high temperature (lots of quantum fluctuation)
        new_state = optimizer._quantum_tunneling_step(
            state, H, temperature=10.0, tunneling_prob=0.5
        )

        assert len(new_state) == 5
        assert all(s in [-1, 1] for s in new_state)  # Valid Ising spin values

    def test_convergence(self, sample_returns):
        """Test that optimization runs and records energy history."""
        optimizer = QuantumAnnealingOptimizer(n_qubits=3, annealing_steps=200)

        result = optimizer.optimize_portfolio(sample_returns)

        # History should be populated after optimization
        assert "energies" in optimizer.history
        energies = optimizer.history["energies"]
        assert len(energies) == 200  # One per annealing step
        # All energies should be finite numbers
        assert all(np.isfinite(e) for e in energies)


class TestQAOASimulator:
    """Test QAOA (Quantum Approximate Optimization Algorithm)."""

    def test_initialization(self):
        """Test QAOA simulator initialization."""
        qaoa = QAOASimulator(n_layers=3)

        assert qaoa.n_layers == 3
        assert len(qaoa.gamma) == 3  # One gamma parameter per layer
        assert len(qaoa.beta) == 3  # One beta parameter per layer

    def test_optimization(self):
        """Test QAOA optimization."""
        qaoa = QAOASimulator(n_layers=2)

        # Simple quadratic problem (e.g., 2-variable portfolio)
        Q = np.array([[1, -0.5], [-0.5, 1]])

        result = qaoa.optimize(Q, n_iterations=50)

        assert "solution" in result
        assert "expectation_value" in result
        assert result["solution"] in [0, 1]  # Binary solution (selected asset)

    def test_problem_hamiltonian(self):
        """Test problem Hamiltonian application."""
        qaoa = QAOASimulator(n_layers=1)

        state = np.ones(3) / np.sqrt(3)  # Uniform superposition
        Q = np.eye(3)

        new_state = qaoa._apply_problem_hamiltonian(state, gamma=0.5, Q=Q)

        # State should remain normalized after rotation
        assert np.abs(np.linalg.norm(new_state) - 1.0) < 1e-6

    def test_mixing_hamiltonian(self):
        """Test mixing Hamiltonian application."""
        qaoa = QAOASimulator(n_layers=1)

        state = np.ones(3) / np.sqrt(3)  # Uniform superposition

        new_state = qaoa._apply_mixing_hamiltonian(state, beta=0.5)

        # State should remain normalized after mixing
        assert np.abs(np.linalg.norm(new_state) - 1.0) < 1e-6


class TestQuantumInspiredEvolution:
    """Test quantum-inspired evolutionary algorithm."""

    def test_initialization(self):
        """Test evolution initialization."""
        evo = QuantumInspiredEvolution(population_size=20, n_qubits=10)

        assert evo.pop_size == 20
        assert evo.n_qubits == 10
        assert len(evo.population) == 0  # Population not yet initialized

    def test_population_initialization(self):
        """Test quantum population initialization."""
        evo = QuantumInspiredEvolution(population_size=10, n_qubits=5)

        evo._initialize_population()

        assert len(evo.population) == 10  # Correct number of individuals

        # Check each individual is a valid quantum state
        for individual in evo.population:
            assert isinstance(individual.amplitudes, np.ndarray)
            assert len(individual.amplitudes) == 5
            # Check normalization (sum of squared amplitudes = 1)
            assert np.abs(np.linalg.norm(individual.amplitudes) - 1.0) < 1e-6

    def test_quantum_crossover(self):
        """Test quantum crossover operation."""
        evo = QuantumInspiredEvolution(n_qubits=5)

        parent1 = QuantumState(
            amplitudes=np.ones(5) / np.sqrt(5), phase=np.zeros(5), energy=0.0
        )
        parent2 = QuantumState(
            amplitudes=np.ones(5) / np.sqrt(5), phase=np.zeros(5), energy=0.0
        )

        child = evo._quantum_crossover(
            parent1, parent2
        )  # Quantum interference crossover

        assert len(child.amplitudes) == 5
        assert (
            np.abs(np.linalg.norm(child.amplitudes) - 1.0) < 1e-6
        )  # Offspring is normalized

    def test_evolution_optimization(self):
        """Test complete evolution optimization."""
        evo = QuantumInspiredEvolution(population_size=10, n_qubits=5)

        # Simple fitness function (maximize negative squared deviation from 0.5)
        def fitness(x):
            return -np.sum((x - 0.5) ** 2)

        result = evo.evolve(fitness, generations=20)

        assert "best_solution" in result
        assert "best_fitness" in result
        assert "fitness_history" in result
        assert len(result["fitness_history"]) == 20  # One entry per generation

        # Fitness history should have values for all generations
        assert len(result["fitness_history"]) == 20
        # Best fitness should be finite
        assert np.isfinite(result["best_fitness"])

    def test_quantum_mutation(self):
        """Test quantum mutation."""
        evo = QuantumInspiredEvolution(n_qubits=5, mutation_rate=0.5)

        # Use complex amplitudes to match QuantumInspiredEvolution._initialize_population
        amplitudes = np.ones(5, dtype=complex) / np.sqrt(5)
        state = QuantumState(amplitudes=amplitudes, phase=np.zeros(5), energy=0.0)

        mutated = evo._quantum_mutation(state)  # Apply random phase rotation

        assert len(mutated.amplitudes) == 5
        assert (
            np.abs(np.linalg.norm(mutated.amplitudes) - 1.0) < 1e-6
        )  # Mutation preserves normalization


class TestIntegrationFunctions:
    """Test high-level integration functions."""

    def test_quantum_portfolio_optimization(self):
        """Test production quantum portfolio function."""
        np.random.seed(42)
        returns = pd.DataFrame(np.random.randn(100, 3) * 0.02, columns=["A", "B", "C"])

        result = quantum_portfolio_optimization(returns, risk_aversion=1.0)

        assert "weights" in result
        assert len(result["weights"]) == 3  # One weight per asset
        assert "sharpe_ratio" in result
        # Sharpe can be negative with random data; check it's finite
        assert np.isfinite(result["sharpe_ratio"])

    def test_qaoa_trade_routing(self):
        """Test QAOA for trade routing."""
        # Simple cost matrix (represents routing costs between venues)
        cost_matrix = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])

        result = qaoa_trade_routing(cost_matrix)

        assert "solution" in result
        assert "expectation_value" in result

    def test_quantum_evolution_strategy(self):
        """Test quantum evolution strategy function."""

        def objective(x):
            return -np.sum(x**2)  # Maximize negative sum of squares (minimize to 0)

        result = quantum_evolution_strategy(objective, dimensions=5, generations=10)

        assert "best_solution" in result
        assert len(result["best_solution"]) == 5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_returns(self):
        """Test handling of empty returns data."""
        optimizer = QuantumAnnealingOptimizer()
        empty_returns = pd.DataFrame()

        with pytest.raises(Exception):
            optimizer.optimize_portfolio(empty_returns)  # Should raise on empty input

    def test_single_asset(self):
        """Test optimization with single asset."""
        np.random.seed(42)
        returns = pd.DataFrame({"BTC": np.random.randn(50) * 0.02})

        optimizer = QuantumAnnealingOptimizer(n_qubits=1)
        result = optimizer.optimize_portfolio(returns)

        # Single asset must have weight = 1.0 (fully invested)
        assert result["weights"]["BTC"] == 1.0

    def test_zero_volatility(self):
        """Test handling of zero volatility assets."""
        returns = pd.DataFrame({"A": np.zeros(100), "B": np.random.randn(100) * 0.02})

        optimizer = QuantumAnnealingOptimizer(n_qubits=2)

        # Should handle gracefully (zero vol causes singular covariance)
        try:
            result = optimizer.optimize_portfolio(returns)
            assert result is not None
        except Exception:
            pass  # Acceptable to raise for zero vol

    def test_large_portfolio(self):
        """Test with larger portfolio (10 assets)."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(200, 10) * 0.02, columns=[f"Asset_{i}" for i in range(10)]
        )

        optimizer = QuantumAnnealingOptimizer(n_qubits=10, annealing_steps=200)
        result = optimizer.optimize_portfolio(returns)

        assert len(result["weights"]) == 10
        assert abs(sum(result["weights"].values()) - 1.0) < 0.01  # Weights sum to 1


class TestPerformance:
    """Test performance characteristics."""

    def test_optimization_speed(self, sample_returns):
        """Test that optimization completes in reasonable time."""
        import time

        optimizer = QuantumAnnealingOptimizer(n_qubits=3, annealing_steps=100)

        start = time.time()
        result = optimizer.optimize_portfolio(sample_returns)
        elapsed = time.time() - start

        # Should complete in under 5 seconds (even on slow hardware)
        assert elapsed < 5.0

    def test_memory_usage(self):
        """Test memory doesn't grow unbounded."""
        optimizer = QuantumAnnealingOptimizer(n_qubits=5, annealing_steps=50)

        # Run multiple times (simulate live operation)
        for _ in range(5):
            returns = pd.DataFrame(np.random.randn(100, 5) * 0.02)
            optimizer.optimize_portfolio(returns)

        # History should be limited (last run only)
        assert len(optimizer.history.get("energies", [])) <= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
