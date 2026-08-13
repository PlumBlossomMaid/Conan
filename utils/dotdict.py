"""DotDict — dict with attribute access + Rich pretty printing.

Port from HiddenSinger (如来唱) project.

Usage:
    cfg = DotDict(yaml.load(open("config.yaml")))
    print(cfg.training.batch_size)  # attribute access
    cfg.print()                     # pretty print
"""
import json
import math

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.syntax import Syntax

from pygments.token import Token
from pygments.style import Style


class DotDict(dict):
    """dict subclass with attribute access + recursive DotDict for nested dicts."""

    def __init__(self, kwargs=None):
        if kwargs is None:
            kwargs = {}
        for k, v in kwargs.items():
            if type(v) is dict:
                v = DotDict(v)
            self[k] = v

    def __getattr__(self, key):
        if key.startswith("_"):
            return super().__getattribute__(key)
        try:
            val = self[key]
            return DotDict(val) if type(val) is dict else val
        except KeyError:
            raise AttributeError(f"DotDict has no key '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"DotDict has no key '{key}'")

    @property
    def output(self):
        """Default output settings — bright colors, bold, dict style."""
        return type("_Output", (), {"bright": True, "bold": True, "style": "dict"})()

    # ── Pretty Print — Rich JSON (dict style) ──

    def print_dict(self):
        """Print config as syntax-highlighted JSON."""
        console = Console()
        plain = dict(self)

        class _JSONStyle(Style):
            styles = {
                Token.Name.Tag: "bold #00ffff",
                Token.Literal.String.Double: "bold #00ff00",
                Token.Literal.Number: "bold #ffff00",
                Token.Keyword.Constant: "bold #ff00ff",
                Token.Punctuation: "bold #ffffff",
            }

        console.print(Syntax(
            json.dumps(plain, ensure_ascii=False, indent=2),
            "json", word_wrap=True, background_color="default",
            theme=_JSONStyle,
        ))

    # ── Pretty Print — Rich Tables (rich style) ──

    def print(self, num_cols=3, max_depth=None):
        """Print config as Rich tables (colored, multi-column)."""
        console = Console()
        terminal_width = console.size.width
        task_name = self.get("task", "Config")
        console.print(f"\n[bold cyan]{task_name}[/bold cyan]\n")

        top_keys = list(self.keys())
        keys_per_col = math.ceil(len(top_keys) / num_cols)
        columns = []

        for col in range(num_cols):
            start = col * keys_per_col
            end = min(start + keys_per_col, len(top_keys))
            col_keys = top_keys[start:end]
            if not col_keys:
                continue

            table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
            table.add_column("Key", style="bold cyan", no_wrap=False, width=24)
            table.add_column("Value", style="bold white", no_wrap=False)

            for key in col_keys:
                self._add_row(table, key, self[key], depth=0, max_depth=max_depth)

            columns.append(Panel(
                table, border_style="dim",
                width=terminal_width // num_cols - 1, padding=(0, 1),
            ))

        console.print(Columns(columns, equal=True, expand=True))
        console.print()

    def _add_row(self, table, key, value, depth=0, max_depth=None, prefix=""):
        indent = "  " * depth
        if isinstance(value, (dict, DotDict)):
            table.add_row(f"{indent}{key}", "")
            for sk, sv in value.items():
                self._add_row(table, sk, sv, depth + 1, max_depth,
                              f"{prefix}.{key}" if prefix else key)
        elif isinstance(value, list):
            if len(value) > 10:
                lines = "\n".join(f"    {i}: {v}" for i, v in enumerate(value))
                table.add_row(f"{indent}{key}", f"[bold yellow]{lines}[/bold yellow]")
            else:
                table.add_row(f"{indent}{key}", f"[bold yellow]{value}[/bold yellow]")
        elif isinstance(value, bool):
            color = "bold green" if value else "bold red"
            table.add_row(f"{indent}{key}", f"[{color}]{value}[/{color}]")
        elif isinstance(value, (int, float)):
            table.add_row(f"{indent}{key}", f"[bold yellow]{value}[/bold yellow]")
        elif isinstance(value, str):
            color = "bold magenta" if "/" in value else "bold green"
            table.add_row(f"{indent}{key}", f"[{color}]'{value}'[/{color}]")
        elif value is None:
            table.add_row(f"{indent}{key}", "[dim]None[/dim]")
        else:
            table.add_row(f"{indent}{key}", str(value))
