#!/usr/bin/env python3
"""
Simple example of loading a pretrained Metamon model and evaluating it against baselines.

Prerequisites:
- Pokemon Showdown server running locally (cd server/pokemon-showdown && node pokemon-showdown start --no-security)
- Environment variable set: export METAMON_CACHE_DIR=/path/to/storage
"""

import functools
import metamon
from metamon.rl.pretrained import get_pretrained_model
from metamon.env import BattleAgainstBaseline, get_metamon_teams
from metamon.baselines import get_baseline
from metamon.rl.metamon_to_amago import MetamonAMAGOWrapper
from metamon.rl.evaluate import pretrained_vs_pokeagent_ladder
import numpy as np
import os

def evaluate_pretrained_model(
    model_name="SmallRL",
    opponent_baseline="RandomBaseline",
    battle_format="gen9ou",
    num_battles=10
):
    """
    Load a pretrained model and evaluate it against a baseline opponent.

    Args:
        model_name: Name of pretrained model (e.g., "SmallRL", "MediumRL", "SyntheticRLV2")
        opponent_baseline: Baseline opponent ("RandomBaseline", "MaxBPBaseline", "GymLeader", "Grunt")
        battle_format: Format to play (e.g., "gen1ou", "gen2ou", "gen3ou", "gen4ou")
        num_battles: Number of battles to run
    """

    print(f"\n=== Evaluating {model_name} vs {opponent_baseline} in {battle_format} ===\n")

    # Step 1: Load the pretrained model
    print(f"Loading pretrained model: {model_name}...")
    model = get_pretrained_model(model_name)
    experiment = model.initialize_agent()

    # Step 2: Create the environment with the baseline opponent
    print(f"Setting up battle environment against {opponent_baseline}...")

    # Get baseline class by name
    baseline_cls = get_baseline(opponent_baseline)

    # Get team set for battles (using competitive teams by default)
    team_set = get_metamon_teams(battle_format, "competitive")

    # Create environment factory
    def make_env():
        env = BattleAgainstBaseline(
            battle_format=battle_format,
            observation_space=model.observation_space,
            action_space=model.action_space,
            reward_function=model.reward_function,
            team_set=team_set,
            opponent_type=baseline_cls,
            turn_limit=200,
            battle_backend="poke-env",
        )
        # Wrap for AMAGO compatibility
        return MetamonAMAGOWrapper(env)



    # Step 3: Use AMAGO's evaluation loop
    print(f"Running {num_battles} battles...\n")

    # Configure experiment for single-env evaluation
    experiment.env_mode = "sync"
    experiment.parallel_actors = 1
    experiment.verbose = False

    # Run evaluation using AMAGO's built-in evaluation loop
    # We estimate ~250 timesteps per battle
    results = experiment.evaluate_test(
        make_test_env=make_env,
        timesteps=num_battles * 250,
        episodes=num_battles,  # Stop after this many episodes
    )

    # Step 4: Report results
    print(f"\n=== Results ===")

    # Extract key metrics from results
    avg_return = results.get("Average Total Return (Across All Env Names)", 0)

    # Estimate win rate from return (positive return usually means win in default reward function)
    if "Win Rate" in results:
        win_rate = results["Win Rate"] * 100
        print(f"Win Rate: {win_rate:.1f}%")

    print(f"Average Return: {avg_return:.2f}")

    # Print any environment-specific metrics
    for key, value in results.items():
        if "Env:" in key:
            print(f"{key}: {value:.2f}")

    return results


def evaluate_on_pokechamp_ladder(
    model_name="SmallRL",
    battle_format="gen9ou",
    num_battles=10
):
    """
    Load a pretrained model and evaluate it on the Pokechamp ladder.

    Args:
        model_name: Name of pretrained model (e.g., "SmallRL", "MediumRL", "SyntheticRLV2")
        battle_format: Format to play (e.g., "gen1ou", "gen2ou", "gen3ou", "gen4ou")
        num_battles: Number of battles to run
    """

    print(f"\n=== Evaluating {model_name} on Pokechamp Ladder in {battle_format} ===\n")

    # Step 1: Check for required credentials
    username = "PAC-MetaHorns"
    password = os.getenv("PAC_PASSWORD")
    if password is None:
        raise RuntimeError("PAC_PASSWORD environment variable is not set")

    # Step 2: Load the pretrained model
    print(f"Loading pretrained model: {model_name}...")
    model = get_pretrained_model(model_name)

    # Step 3: Evaluate on the Pokechamp ladder
    print(f"Connecting to Pokechamp ladder with username: {username}...")
    print(f"Running {num_battles} battles...\n")

    # Get team set for battles
    team_set = get_metamon_teams(battle_format, "competitive")

    # Run evaluation on the ladder
    results = pretrained_vs_pokeagent_ladder(
        pretrained_model=model,
        username=username,
        password=password,
        battle_format=battle_format,
        team_set=team_set,
        total_battles=num_battles,
        battle_backend="poke-env",
    )

    # Step 4: Report results
    print(f"\n=== Ladder Results ===")

    if isinstance(results, dict):
        for key, value in results.items():
            print(f"{key}: {value}")
    else:
        print(f"Results: {results}")

    return results


def main():
    """
    Run a few example evaluations with different models and opponents.
    Supports both baseline evaluations and Pokechamp ladder evaluations.
    """

    # Choose evaluation type: "baseline" or "ladder"
    evaluation_type = "ladder"  # Change to "ladder" to evaluate on Pokechamp ladder

    if evaluation_type == "baseline":
        # === Baseline Evaluations ===

        # Example 1: Small model vs random baseline (should win almost always)
        # evaluate_pretrained_model(
        #     model_name="SmallRL",
        #     opponent_baseline="RandomBaseline",
        #     battle_format="gen9ou",
        #     num_battles=5
        # )

        # Example 2: Small model vs stronger baseline
        # evaluate_pretrained_model(
        #     model_name="SmallRL",
        #     opponent_baseline="MaxBPBaseline",
        #     battle_format="gen9ou",
        #     num_battles=5
        # )

        # Example 3: Stronger model vs GymLeader (will download ~1GB)
        evaluate_pretrained_model(
            model_name="SyntheticRLV2",
            opponent_baseline="GymLeader",
            battle_format="gen9ou",
            num_battles=100
        )

    elif evaluation_type == "ladder":
        # === Pokechamp Ladder Evaluations ===
        # Note: Requires PAC_PASSWORD environment variable to be set

        # Example 1: Small model on ladder
        # evaluate_on_pokechamp_ladder(
        #     model_name="SmallRL",
        #     battle_format="gen9ou",
        #     num_battles=5
        # )

        # Example 2: Stronger model on ladder
        evaluate_on_pokechamp_ladder(
            model_name="SyntheticRLV2",
            battle_format="gen9ou",
            num_battles=100
        )

    else:
        print(f"Unknown evaluation type: {evaluation_type}")
        print("Please use 'baseline' or 'ladder'")


if __name__ == "__main__":
    # Check that cache dir is set
    if metamon.METAMON_CACHE_DIR is None:
        print("Error: Please set METAMON_CACHE_DIR environment variable")
        print("Example: export METAMON_CACHE_DIR=/tmp/metamon_cache")
        exit(1)

    # Run the examples
    main()