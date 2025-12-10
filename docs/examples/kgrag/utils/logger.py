import asyncio
import logging
import json
import os
import time
from typing import Any, Dict

old_factory = logging.getLogRecordFactory()
# Inject task name
def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    try:
        current_task = asyncio.current_task()
    except RuntimeError:
        # No running event loop, use a default value.
        current_task = None
    record.taskName = current_task.get_name() if current_task else "MainThread"
    return record

logging.setLogRecordFactory(record_factory)

# ANSI escape codes for color
LOG_COLORS = {
    'DEBUG': '\033[94m',     # Blue
    'INFO': '\033[92m',      # Green
    'WARNING': '\033[93m',   # Yellow
    'ERROR': '\033[91m',     # Red
    'CRITICAL': '\033[95m',  # Magenta
    'RESET': '\033[0m'       # Reset color
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        log_color = LOG_COLORS.get(record.levelname, LOG_COLORS['RESET'])
        reset = LOG_COLORS['RESET']
        record.levelname = f"{log_color}{record.levelname}{reset}"
        record.msg = f"{log_color}{record.msg}{reset}"
        return super().format(record)

class BaseProgressLogger(logging.Logger):
    """
    A base logger class that extends the standard Python Logger to support progress tracking.

    It handles loading, saving, and updating progress data in a JSON file and is designed
    to be subclassed for domain-specific logging (e.g., KG updates, QA evaluation).

    Attributes:
        progress_path (str): Path to the JSON file where progress data is stored.
        progress_data (Dict[str, Any]): In-memory dictionary tracking progress.
    """

    def __init__(
        self,
        name: str,
        progress_path: str,
        default_progress_data: Dict[str, Any],
        level: int = logging.DEBUG
    ):
        """
        Initializes the BaseProgressLogger.

        Args:
            name (str): Name of the logger.
            progress_path (str): File path for saving progress JSON.
            default_progress_data (Dict[str, Any]): Default structure for progress data.
            level (int, optional): Logging level. Defaults to logging.DEBUG.
        """
        super().__init__(name, level)
        self.progress_path = progress_path
        self.progress_data = default_progress_data.copy()

        # Optional: add a default stream handler
        if not self.handlers:
            handler = logging.StreamHandler()
            formatter = ColorFormatter(
                '%(asctime)s | %(levelname)-7s | %(taskName)s | %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
            handler.setFormatter(formatter)
            self.addHandler(handler)

        self.load_progress()
        self.processed = set([stat.get("id") for stat in self.progress_data.get("stats", [])])

    def load_progress(self):
        """
        Loads progress data from the progress_path JSON file.
        Falls back to default if the file does not exist or cannot be parsed.
        """
        if os.path.exists(self.progress_path):
            try:
                with open(self.progress_path, "r", encoding="utf-8") as f:
                    self.progress_data = json.load(f)
                self.info(f"Loaded progress from {self.progress_path}")
            except Exception as e:
                self.warning(f"Failed to load progress", exc_info=True)
        else:
            self.info("No previous progress found. Starting fresh.")

    def save_progress(self, max_retries = 10):
        """
        Saves the current progress_data to the JSON file at progress_path.
        """
        for attempt in range(max_retries):
            try:
                with open(self.progress_path, "w", encoding="utf-8") as f:
                    json.dump(self.progress_data, f, indent=4, ensure_ascii=False)
                self.debug("Progress saved.")
                return
            except Exception as e:
                self.error(f"[Retry {attempt+1}/{max_retries}] Failed to save progress", exc_info=True)
                time.sleep(min(2 ** attempt, 60))  # Exponential backoff (2s, 4s, 8s, etc.)
        raise Exception("Failed to save the latest progress!")

    def update_progress(self, pairs: dict):
        """
        Updates one or more key-value pairs in progress_data and saves.

        Args:
            pairs (dict): Dictionary of progress values to update.
        """
        self.progress_data.update(pairs)
        self.debug(f"Progress updated: {pairs}")
        self.save_progress()

    def add_stat(self, stat: dict):
        """
        Appends a statistic entry to the 'stats' list in progress_data.

        Args:
            stat (dict): Statistic entry to append.
        """
        self.progress_data.setdefault("stats", []).append(stat)
        self.processed.add(stat.get("id"))
        self.debug(f"Added stat: {stat}")
        self.save_progress()

class DefaultProgressLogger(BaseProgressLogger):
    """
    A generic progress logger used for debugging, development,
    or as a default logger when none is provided.

    This logger stores minimal progress data and is safe to use
    in utility functions or scripts that optionally accept a logger.
    """

    def __init__(self, name: str = "DefaultProgressLogger"):
        """
        Initializes the default progress logger with in-memory progress data.

        Args:
            name (str): Logger name. Defaults to "DefaultProgressLogger".
        """
        # Use an in-memory dummy path to avoid saving to disk
        dummy_path = os.devnull  # Cross-platform null device
        default_data = {
            "note": "This logger is used for development/debugging only.",
            "stats": []
        }
        super().__init__(name, dummy_path, default_data)

    def save_progress(self):
        """
        Overrides save_progress to avoid writing to disk.
        """
        # self.debug("(Skipping save) This is a default in-memory progress logger.")
        pass

    def load_progress(self):
        """
        Overrides load_progress to avoid reading from disk.
        """
        # self.debug("(Skipping load) This is a default in-memory progress logger.")
        pass

class KGProgressLogger(BaseProgressLogger):
    """
    Logger subclass for tracking knowledge graph (KG) update progress.
    """

    def __init__(self, progress_path: str):
        """
        Initializes KGProgressLogger with KG-specific progress structure.

        Args:
            progress_path (str): File path for storing progress data.
        """
        default_data = {
            "last_doc_total": None,
            "stats": []
        }
        super().__init__("KGLogger", progress_path, default_data)

    @property
    def processed_docs(self) -> int:
        """
        Returns a set of processed document IDs.

        Returns:
            int: Count of processed documents.
        """
        return self.processed
    

class QAProgressLogger(BaseProgressLogger):
    """
    Logger subclass for tracking open-domain QA inference progress and logs.
    """

    def __init__(self, progress_path: str):
        """
        Initializes QAProgressLogger with QA-specific progress structure.

        Args:
            progress_path (str): File path for storing progress data.
        """
        default_data = {
            "last_question_total": 0,
            "stats": []
        }
        super().__init__("QALogger", progress_path, default_data)

    @property
    def processed_questions(self) -> int:
        """
        Returns a set of processed question IDs.

        Returns:
            int: Count of processed questions.
        """
        return self.processed

    def add_qa_log(self, log: dict):
        """
        Appends a QA log entry (query, prediction, etc.) to the log list.

        Args:
            log (dict): QA log entry to append.
        """
        self.progress_data["qa_logs"].append(log)
        self.debug(f"Added QA log: {log}")
        self.save_progress()

logger = DefaultProgressLogger()