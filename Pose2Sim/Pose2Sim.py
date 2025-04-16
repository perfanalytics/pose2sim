#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
###########################################################################
## POSE2SIM                                                              ##
###########################################################################

This repository offers a way to perform markerless kinematics, and gives an
example workflow from an Openpose input to an OpenSim result.

It offers tools for:
- Cameras calibration,
- 2D pose estimation,
- Camera synchronization,
- Tracking the person of interest,
- Robust triangulation,
- Filtration,
- Marker augmentation,
- OpenSim scaling and inverse kinematics

It has been tested on Windows, Linux and MacOS, and works for any Python version >= 3.9

Installation:
# Open Anaconda prompt. Type:
# - conda create -n Pose2Sim python=3.9
# - conda activate Pose2Sim
# - conda install -c opensim-org opensim -y
# - pip install Pose2Sim

Usage:
# First run Pose estimation and organize your directories (see Readme.md)
from Pose2Sim import Pose2Sim
Pose2Sim.calibration()
Pose2Sim.poseEstimation()
Pose2Sim.synchronization()
Pose2Sim.personAssociation()
Pose2Sim.triangulation()
Pose2Sim.filtering()
Pose2Sim.markerAugmentation()
Pose2Sim.kinematics()
# Then run OpenSim (see Readme.md)
'''

import importlib
import os
import time
import logging
from datetime import datetime
from typing import Any, Iterable, Sequence

from Pose2Sim.config import Config
from Pose2Sim.stages.base import BaseStage

# AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"

def to_camel_case(snake_str):
    return ''.join(word.capitalize() for word in snake_str.split('_'))

def _discover_stage(name: str) -> type[BaseStage]:
    from importlib import metadata

    for ep in metadata.entry_points(group="pose2sim.stages"):
        if ep.name == name:
            cls = ep.load()
            if not issubclass(cls, BaseStage):
                raise TypeError(f"Entry‑point '{name}' is not a BaseStage subclass")
            return cls

    try:
        mod = importlib.import_module(f"Pose2Sim.stages.{name}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Stage '{name}' not found (no entry-point, no module)") from exc

    camel = f"{to_camel_case(name)}Stage"
    if not hasattr(mod, camel):
        raise AttributeError(f"Module '{mod.__name__}' has no class '{camel}'")

    cls = getattr(mod, camel)
    if not issubclass(cls, BaseStage):
        raise TypeError(f"{camel} is not a BaseStage subclass")
    return cls


# ────────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ────────────────────────────────────────────────────────────────────────────────


class Pose2SimPipeline:
    """Light orchestration: chain the requested stages.

    Parameters
    ----------
    config : str | os.PathLike | dict | None
        Path to a TOML/JSON/YAML config or a dict already loaded.
    stages : Sequence[str] | None
        Ordered list of stage *names* to execute. ``None`` = default full chain.
    log_level : str
        Logging level ("INFO", "DEBUG", …).
    """

    DEFAULT_CHAIN: tuple[str, ...] = (
        "calibration",
        "pose_estimation",
        "sync",
        "tracking",
        "triangulation",
        "filtering",
        "markeraugmentation",
        "ik",
    )

    # ---------------------------------------------------------------------
    #  Init / context‑manager helpers
    # ---------------------------------------------------------------------

    def __init__(
        self,
        config_input: str | os.PathLike | dict | None = None,
        stages: Sequence[str] | None = None,
        *,
        log_level: str = "INFO",
    ) -> None:
        self.config = Config(config_input)

        if not self.config.use_custom_logging:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(
                format='%(message)s',
                level=logging.INFO,
                handlers=[
                    logging.handlers.TimedRotatingFileHandler(
                        os.path.join(self.config.session_dir, 'logs.txt'),
                        when='D',
                        interval=7
                    ),
                    logging.StreamHandler()
                ]
            )

        # ––– Instantiate stages –––––––––––––––––––––––––––––––––––––––––
        wanted = stages or self.DEFAULT_CHAIN
        self._stages: list[BaseStage] = [
            _discover_stage(name)(self.config) for name in wanted
        ]

    # ------------------------------------------------------------------
    #  Context‑manager so we guarantee setup/teardown
    # ------------------------------------------------------------------

    def __enter__(self) -> "Pose2SimPipeline":
        for st in self._stages:
            st.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for st in reversed(self._stages):
            st.teardown()
        logging.info("Pose2Sim pipeline completed.")

    # ------------------------------------------------------------------
    #  Execution
    # ------------------------------------------------------------------

    def run(self, initial_data: Any = None) -> Any:
        """Run the chain and return the last stage output."""
        data = initial_data
        for st in self._stages:
            t0 = time.time()
            logging.info("\n──────────────────────────────────────────────────────────────")
            logging.info(
                f"{st.name.upper()}  |  {datetime.now().strftime('%H:%M:%S')}  |  START"
            )
            data = st.run(data)
            logging.info(
                f"{st.name.upper()}  |  done in {time.time() - t0:.2f} s"
            )
        return data

    @classmethod
    def _single(cls, stage: str, cfg: str | os.PathLike | dict | None = None):
        with cls(cfg, stages=[stage]) as pipe:
            pipe.run()


calibration = lambda cfg=None: Pose2SimPipeline._single("calibration", cfg)
poseEstimation = lambda cfg=None: Pose2SimPipeline._single("pose_estimation", cfg)
synchronization = lambda cfg=None: Pose2SimPipeline._single("sync", cfg)
personAssociation = lambda cfg=None: Pose2SimPipeline._single("tracking", cfg)
triangulation = lambda cfg=None: Pose2SimPipeline._single("triangulation", cfg)
filtering = lambda cfg=None: Pose2SimPipeline._single("filtering", cfg)
markerAugmentation = lambda cfg=None: Pose2SimPipeline._single("markeraugmentation", cfg)
kinematics = lambda cfg=None: Pose2SimPipeline._single("ik", cfg)


def runAll(cfg: str | os.PathLike | dict | None = None, stages: Iterable[str] | None = None):
    """Run the whole chain (or the *stages* list)."""

    with Pose2SimPipeline(cfg, stages=stages) as pipe:
        pipe.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pose2Sim – modular pipeline")
    parser.add_argument("config", nargs="?", help="Path to Config.toml (or JSON/YAML)")
    parser.add_argument(
        "--stages",
        nargs="*",
        help="Subset of stages to run (order preserved). Default = full chain.",
    )
    args = parser.parse_args()

    runAll(args.config, stages=args.stages)
