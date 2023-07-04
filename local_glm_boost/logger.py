import logging
from typing import Optional
import os

import numpy as np


class LocalGLMBoostLogger:
    """Logger for the simulation study."""

    def __init__(
        self,
        verbose: int = 0,
        output_path: Optional[str] = None,
    ):
        """Initialize the logger.

        :param verbose: The verbosity level.
        :param output_path: The path to the output directory.
        """
        self.verbose = verbose
        self.last_progress = 0
        self.logger = logging.Logger("simulation_logger")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        formatter = logging.Formatter(
            f"[%(asctime)s][%(message)s]",
            datefmt="%Y-%m-%d %H:%M",
        )
        self.logger.handlers[0].setFormatter(formatter)

        if output_path is not None:
            log_file = os.path.join(output_path, "log.txt")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.handlers[1].setFormatter(formatter)

    def log(self, msg: str, verbose: int = 0):
        if verbose <= self.verbose:
            self.logger.info(msg)

    def log_progress(self, step: int, total_steps: int, verbose: int = 0):
        """Log the progress of the simulation.

        :param step: The current step.
        :param total_steps: The total number of steps.
        :param verbose: The verbosity level.
        """
        # Check if progress has been made
        new_progress = np.floor(10 * step / total_steps) / 10
        if new_progress > self.last_progress:
            self.last_progress = new_progress
            msg = f"progress: {int(new_progress * 100)}%"

            if verbose <= self.verbose:
                self.logger.info(msg)

    def reset_progress(self):
        """Reset the progress."""
        self.last_progress = 0

    def append_format_level(self, level_msg):
        """Append the level to the message.

        :param level_msg: The level to append to the message.
        """
        formatter = self.logger.handlers[0].formatter
        format_msg = formatter._fmt.split("[%(message)s]")[0]
        format_msg += f"[{level_msg}][%(message)s]"
        formatter = logging.Formatter(format_msg, datefmt="%Y-%m-%d %H:%M")
        self.logger.handlers[0].setFormatter(formatter)
        if len(self.logger.handlers) > 1:
            self.logger.handlers[1].setFormatter(formatter)

    def remove_format_level(self):
        """Remove one level from the message."""
        formatter = self.logger.handlers[0].formatter
        format_msg = formatter._fmt.rsplit("[", 2)[0]
        format_msg += "[%(message)s]"
        formatter = logging.Formatter(format_msg, datefmt="%Y-%m-%d %H:%M")
        self.logger.handlers[0].setFormatter(formatter)
        if len(self.logger.handlers) > 1:
            self.logger.handlers[1].setFormatter(formatter)
