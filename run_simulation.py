#!/usr/bin/env python3
"""
Entry point for the traffic simulation.

Usage:
    python run_simulation.py [--traffic 0.5] [--speed 1.0] [--cols 3] [--rows 3]

Controls:
    ↑/↓    Increase/decrease traffic level
    +/-    Increase/decrease simulation speed
    P      Pause/resume
    R      Reset simulation
    ESC    Quit
"""
import argparse
import sys

import pygame

from src.simulation.config import SimulationConfig
from src.simulation.cities.grid_city import build_grid_city
from src.simulation.simulator import Simulator
from src.simulation.renderer import Renderer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semáforos IA — Traffic Simulation")
    p.add_argument("--traffic", type=float, default=0.5, help="Traffic level 0.0–1.0")
    p.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier")
    p.add_argument("--cols", type=int, default=4, help="Grid columns")
    p.add_argument("--rows", type=int, default=4, help="Grid rows")
    p.add_argument("--block", type=float, default=120.0, help="Block size in meters")
    p.add_argument("--max-vehicles", type=int, default=200, help="Max concurrent vehicles")
    return p.parse_args()


def build_simulation(args: argparse.Namespace) -> tuple[Simulator, Renderer]:
    config = SimulationConfig(
        traffic_level=args.traffic,
        time_scale=args.speed,
        max_vehicles=args.max_vehicles,
    )

    city = build_grid_city(
        cols=args.cols,
        rows=args.rows,
        block_size=args.block,
    )

    sim = Simulator(city, config)
    renderer = Renderer(config)
    return sim, renderer


def main() -> None:
    args = parse_args()
    sim, renderer = build_simulation(args)
    paused = False

    try:
        while True:
            dt = renderer.tick()
            sim_dt = dt * sim.config.time_scale

            # --- Events ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        sim, renderer = build_simulation(args)
                        paused = False
                    elif event.key == pygame.K_UP:
                        sim.config.traffic_level = min(1.0, sim.config.traffic_level + 0.1)
                    elif event.key == pygame.K_DOWN:
                        sim.config.traffic_level = max(0.0, sim.config.traffic_level - 0.1)
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        sim.config.time_scale = min(10.0, sim.config.time_scale + 0.5)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        sim.config.time_scale = max(0.0, sim.config.time_scale - 0.5)

            # --- Update ---
            if not paused:
                sim.tick(sim_dt)

            # --- Render ---
            renderer.render(sim)

    except KeyboardInterrupt:
        pass
    finally:
        renderer.destroy()


if __name__ == "__main__":
    main()
