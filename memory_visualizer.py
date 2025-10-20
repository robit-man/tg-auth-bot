#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Visualizer - Curses-based visualization of bot's internal memory states
Displays memory recalls, scopes, and operations in a Game of Life-style ASCII interface
"""

try:
    import curses  # type: ignore
    CURSES_AVAILABLE = True
except Exception:
    curses = None  # type: ignore
    CURSES_AVAILABLE = False
import time
import threading
import queue
import math
import random
import locale
import os
import sys
from collections import deque
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class MemoryVisualizer:
    """Beautiful curses-based memory state visualizer with Game of Life aesthetics"""

    def __init__(self, force_curses=False):
        self.stdscr = None
        self.running = False
        self.use_fallback = False
        self.force_curses = force_curses  # Force curses mode even without TTY
        self.curses_available = CURSES_AVAILABLE
        self.update_queue = queue.Queue(maxsize=1000)
        self.recall_log = deque(maxlen=100)

        # Memory state tracking
        self.memory_grid = {}  # (x, y) -> cell_data
        self.active_memories = {}  # memory_id -> {scope, category, weight, age, pos}
        self.memory_stats = {
            'thread': 0,
            'user': 0,
            'global': 0,
            'total_recalls': 0,
            'last_recall_time': 0,
        }

        # Autonomous operation tracking
        self.autonomous_ops = deque(maxlen=50)  # Recent autonomous operations
        self.autonomous_stats = {
            'rollups': 0,
            'decays': 0,
            'consolidations': 0,
            'assessments': 0,
            'last_rollup_time': 0,
            'last_decay_time': 0,
        }
        self.autonomous_indicator = 0  # Animation counter for autonomous activity

        # Sleep cycle tracking
        self.sleep_state = 'awake'
        self.sleep_time_in_state = 0
        self.sleep_cycle_count = 0
        self.sleep_discoveries = {}

        # Visual state
        self.grid_width = 80
        self.grid_height = 20
        self.animation_frame = 0
        self.color_pairs = {}
        self._last_console_render = 0.0
        self._last_console_render = 0.0

        # Activity patterns (Game of Life style)
        self.activity_cells = set()  # Active cells that "live"
        self.decay_map = {}  # Cell position -> decay counter

    def init_colors(self):
        """Initialize color pairs for different memory types"""
        if not self.curses_available or not curses:
            return
        if not curses.has_colors():
            return

        curses.start_color()
        curses.use_default_colors()

        # Define color pairs for different memory scopes and states
        curses.init_pair(1, curses.COLOR_CYAN, -1)      # Thread memories
        curses.init_pair(2, curses.COLOR_GREEN, -1)     # User memories
        curses.init_pair(3, curses.COLOR_MAGENTA, -1)   # Global memories
        curses.init_pair(4, curses.COLOR_YELLOW, -1)    # Active recall
        curses.init_pair(5, curses.COLOR_RED, -1)       # High weight
        curses.init_pair(6, curses.COLOR_BLUE, -1)      # Metadata
        curses.init_pair(7, curses.COLOR_WHITE, -1)     # Default

        self.color_pairs = {
            'thread': curses.color_pair(1),
            'user': curses.color_pair(2),
            'global': curses.color_pair(3),
            'active': curses.color_pair(4) | curses.A_BOLD,
            'high_weight': curses.color_pair(5) | curses.A_BOLD,
            'metadata': curses.color_pair(6),
            'default': curses.color_pair(7),
        }

    def start(self):
        """Start the curses visualization in a separate thread"""
        if self.running:
            return

        self.running = True

        if not self.curses_available:
            print("[visualizer] curses module not available – using console fallback renderer.")
            self.use_fallback = True
            self.vis_thread = threading.Thread(target=self._run_fallback, daemon=True)
            self.vis_thread.start()
            return

        if not sys.stdout.isatty() and not self.force_curses:
            print("[visualizer] stdout is not a TTY; using console fallback renderer.")
            print("[visualizer] Set FORCE_CURSES=true to force curses mode.")
            self.use_fallback = True
            self.vis_thread = threading.Thread(target=self._run_fallback, daemon=True)
            self.vis_thread.start()
            return

        os.environ.setdefault("TERM", "xterm-256color")
        try:
            locale.setlocale(locale.LC_ALL, "")
        except locale.Error:
            pass

        self.use_fallback = False
        self.vis_thread = threading.Thread(target=self._run_curses, daemon=True)
        self.vis_thread.start()

    def stop(self):
        """Stop the visualization"""
        self.running = False
        if hasattr(self, 'vis_thread'):
            self.vis_thread.join(timeout=2)

    def log_recall(self, scope: str, category: str, content: str, weight: float, metadata: dict = None):
        """Log a memory recall event"""
        timestamp = time.time()
        entry = {
            'time': timestamp,
            'scope': scope,
            'category': category,
            'content': content[:80],
            'weight': weight,
            'metadata': metadata or {}
        }

        try:
            self.update_queue.put_nowait(('recall', entry))
        except queue.Full:
            pass  # Drop if queue is full

    def log_memory_operation(self, operation: str, details: dict):
        """Log a memory operation (store, retrieve, update, etc.)"""
        try:
            self.update_queue.put_nowait(('operation', {
                'op': operation,
                'time': time.time(),
                **details
            }))
        except queue.Full:
            pass

    def update_stats(self, stats: dict):
        """Update memory statistics"""
        try:
            self.update_queue.put_nowait(('stats', stats))
        except queue.Full:
            pass

    def _run_curses(self):
        """Main curses loop"""
        if not self.curses_available or not curses:
            self.use_fallback = True
            self._run_fallback()
            return
        try:
            curses.wrapper(self._curses_main)
        except Exception as e:
            self.running = False
            print(f"[visualizer] Failed to start curses UI: {e}")
            try:
                if curses:
                    curses.endwin()
            except Exception:
                pass
            # Fallback to console mode if possible
            if not self.use_fallback:
                self.use_fallback = True
                self.running = True
                self._run_fallback()

    def _run_fallback(self):
        """Simple console renderer when curses is unavailable"""
        while self.running:
            updated = False
            try:
                msg_type, data = self.update_queue.get(timeout=0.5)
                self._process_update(msg_type, data)
                updated = True
            except queue.Empty:
                pass

            now = time.time()
            if updated or (now - self._last_console_render) > 1.5:
                self._render_console(now)
                self._last_console_render = now

    def _render_console(self, timestamp: float):
        """Render a compact textual snapshot to stdout"""
        if self.recall_log:
            recent = self.recall_log[-1]
            scope = recent.get('scope', '?')
            category = recent.get('category', '?')
            content = recent.get('content', '')
            weight = recent.get('weight', 0.0)
            recall_line = f"{datetime.fromtimestamp(recent['time']).strftime('%H:%M:%S')} {scope}::{category} w={weight:.2f} {content}"
        else:
            recall_line = "(no recalls yet)"

        stats = self.memory_stats
        stats_line = f"thread={stats.get('thread',0)} user={stats.get('user',0)} global={stats.get('global',0)} total_recalls={stats.get('total_recalls',0)}"

        sleep_line = f"sleep_state={self.sleep_state} time_in_state={self.sleep_time_in_state:.0f}s cycles={self.sleep_cycle_count}"

        print(f"[visualizer] {timestamp:.0f} | {recall_line}")
        print(f"[visualizer] stats: {stats_line}")
        print(f"[visualizer] sleep: {sleep_line}")
        if self.autonomous_ops:
            last_auto = self.autonomous_ops[-1]
            auto_line = f"{last_auto.get('type','?')} scope={last_auto.get('scope','?')} count={last_auto.get('count',0)}"
            print(f"[visualizer] last autonomous: {auto_line}")
        print("-" * 60)

    def _curses_main(self, stdscr):
        """Main curses display function"""
        self.stdscr = stdscr
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(50)  # 50ms timeout for getch

        self.init_colors()

        while self.running:
            try:
                # Process update queue
                while not self.update_queue.empty():
                    try:
                        msg_type, data = self.update_queue.get_nowait()
                        self._process_update(msg_type, data)
                    except queue.Empty:
                        break

                # Update display
                self._render_frame()

                # Check for resize
                key = stdscr.getch()
                if key == curses.KEY_RESIZE:
                    self._handle_resize()
                elif key == ord('q') or key == ord('Q'):
                    self.running = False

                # Increment animation frame
                self.animation_frame += 1

                # Small delay
                time.sleep(0.05)

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception:
                # Continue on any error
                pass

    def _process_update(self, msg_type: str, data: dict):
        """Process an update message"""
        if msg_type == 'recall':
            self._handle_recall(data)
        elif msg_type == 'operation':
            self._handle_operation(data)
        elif msg_type == 'stats':
            self.memory_stats.update(data)
        elif msg_type == 'autonomous':
            self._handle_autonomous_operation(data)
        elif msg_type == 'sleep_state':
            self._handle_sleep_state(data)

    def _handle_recall(self, data: dict):
        """Handle a memory recall event"""
        # Add to log
        self.recall_log.append(data)

        # Update stats
        scope = data.get('scope', 'unknown')
        if scope in self.memory_stats:
            self.memory_stats[scope] = self.memory_stats.get(scope, 0) + 1
        self.memory_stats['total_recalls'] += 1
        self.memory_stats['last_recall_time'] = data['time']

        # Add to visual grid (Game of Life style)
        self._spawn_memory_cell(data)

    def _handle_operation(self, data: dict):
        """Handle a memory operation event"""
        op = data.get('op', 'unknown')
        # Could visualize different operations differently
        pass

    def _handle_autonomous_operation(self, data: dict):
        """Handle an autonomous operation event"""
        op_type = data.get('type', 'unknown')
        timestamp = data.get('time', time.time())

        # Update stats
        if op_type == 'rollup':
            self.autonomous_stats['rollups'] += 1
            self.autonomous_stats['last_rollup_time'] = timestamp
        elif op_type == 'decay':
            self.autonomous_stats['decays'] += 1
            self.autonomous_stats['last_decay_time'] = timestamp
        elif op_type == 'consolidation':
            self.autonomous_stats['consolidations'] += 1
        elif op_type == 'assessment':
            self.autonomous_stats['assessments'] += 1

        # Add to autonomous operations log
        self.autonomous_ops.append({
            'type': op_type,
            'time': timestamp,
            'details': data.get('details', ''),
            'scope': data.get('scope', ''),
            'count': data.get('count', 0),
        })

        # Trigger visual indicator
        self.autonomous_indicator = 30  # Flash for 30 frames

        # Spawn special effect cells for autonomous operations
        self._spawn_autonomous_effect(data)

    def _handle_sleep_state(self, data: dict):
        """Handle sleep state update"""
        self.sleep_state = data.get('state', 'awake')
        self.sleep_time_in_state = data.get('time_in_state', 0)
        self.sleep_cycle_count = data.get('cycle_count', 0)
        self.sleep_discoveries = data.get('discoveries', {})

        # Visual indicator for sleep state changes
        if self.sleep_state != 'awake':
            self.autonomous_indicator = max(self.autonomous_indicator, 20)

    def _spawn_memory_cell(self, memory_data: dict):
        """Spawn a new cell in the memory grid based on recall"""
        scope = memory_data.get('scope', 'unknown')
        weight = memory_data.get('weight', 0.5)

        # Determine spawn position based on scope
        if scope == 'thread':
            zone_x = self.grid_width // 4
        elif scope == 'user':
            zone_x = self.grid_width // 2
        elif scope == 'global':
            zone_x = 3 * self.grid_width // 4
        else:
            zone_x = self.grid_width // 2

        # Add some randomness
        x = max(0, min(self.grid_width - 1, zone_x + random.randint(-5, 5)))
        y = random.randint(0, max(0, self.grid_height - 1))

        # Create cell
        cell_key = (x, y)
        self.memory_grid[cell_key] = {
            'scope': scope,
            'weight': weight,
            'age': 0,
            'category': memory_data.get('category', ''),
        }

        # Add to activity set for Conway's Game of Life style animation
        self.activity_cells.add(cell_key)
        self.decay_map[cell_key] = int(weight * 20) + 5  # Decay based on weight

    def _spawn_autonomous_effect(self, data: dict):
        """Spawn visual effects for autonomous operations"""
        op_type = data.get('type', 'unknown')
        scope = data.get('scope', '')
        count = data.get('count', 1)

        # Determine spawn zone based on scope or operation type
        if scope == 'thread':
            zone_x = self.grid_width // 4
        elif scope == 'user':
            zone_x = self.grid_width // 2
        elif scope == 'global':
            zone_x = 3 * self.grid_width // 4
        else:
            # Autonomous ops spawn in center if no specific scope
            zone_x = self.grid_width // 2

        # Spawn multiple cells for larger operations
        num_cells = min(count, 10)
        for _ in range(num_cells):
            x = max(0, min(self.grid_width - 1, zone_x + random.randint(-8, 8)))
            y = random.randint(0, max(0, self.grid_height - 1))

            cell_key = (x, y)
            self.memory_grid[cell_key] = {
                'scope': scope or 'autonomous',
                'weight': 0.9,  # Autonomous ops are high priority
                'age': 0,
                'category': op_type,
                'autonomous': True,  # Mark as autonomous operation
            }

            self.activity_cells.add(cell_key)
            self.decay_map[cell_key] = 15  # Autonomous effects last longer

    def _update_game_of_life(self):
        """Update cells using Game of Life-inspired rules"""
        # Decay existing cells
        to_remove = []
        for cell_pos in list(self.activity_cells):
            if cell_pos in self.decay_map:
                self.decay_map[cell_pos] -= 1
                if self.decay_map[cell_pos] <= 0:
                    to_remove.append(cell_pos)

        for cell_pos in to_remove:
            self.activity_cells.discard(cell_pos)
            self.decay_map.pop(cell_pos, None)
            self.memory_grid.pop(cell_pos, None)

        # Age existing memories
        for cell_data in self.memory_grid.values():
            cell_data['age'] += 1

    def _handle_resize(self):
        """Handle terminal resize"""
        if self.stdscr:
            self.stdscr.clear()

    def _render_frame(self):
        """Render the current frame"""
        if not self.stdscr:
            return

        try:
            height, width = self.stdscr.getmaxyx()
            self.stdscr.clear()

            # Update grid dimensions
            self.grid_width = max(40, width - 2)
            self.grid_height = max(10, height - 8)

            # Update Game of Life cells
            if self.animation_frame % 2 == 0:  # Update every other frame
                self._update_game_of_life()

            # Draw border
            self._draw_border(height, width)

            # Draw title
            self._draw_title(width)

            # Draw memory grid (Game of Life style)
            self._draw_memory_grid(height, width)

            # Draw statistics panel
            self._draw_stats_panel(height, width)

            # Draw recall log at bottom
            self._draw_recall_log(height, width)

            self.stdscr.refresh()

        except curses.error:
            # Ignore curses errors (e.g., writing to edge of screen)
            pass

    def _draw_border(self, height: int, width: int):
        """Draw the main border"""
        try:
            # Top border
            self.stdscr.addstr(0, 0, "+" + "-" * (width - 2) + "+", self.color_pairs.get('metadata', 0))
            # Bottom border
            if height > 1:
                self.stdscr.addstr(height - 1, 0, "+" + "-" * (width - 2) + "+", self.color_pairs.get('metadata', 0))
            # Side borders
            for y in range(1, height - 1):
                self.stdscr.addstr(y, 0, "|", self.color_pairs.get('metadata', 0))
                self.stdscr.addstr(y, width - 1, "|", self.color_pairs.get('metadata', 0))
        except curses.error:
            pass

    def _draw_title(self, width: int):
        """Draw the title bar"""
        try:
            # Sleep state symbols
            sleep_symbols = {
                'awake': '*',
                'drowsy': '~',
                'light_sleep': '-',
                'deep_sleep': '=',
                'rem_sleep': '+',
                'waking': '^'
            }

            symbol = sleep_symbols.get(self.sleep_state, '*')

            if self.sleep_state == 'awake':
                title = f"{symbol} MEMORY CONSCIOUSNESS VISUALIZER {symbol}"
            else:
                state_display = self.sleep_state.upper().replace('_', ' ')
                title = f"{symbol} MEMORY CONSCIOUSNESS [{state_display}] {symbol}"

            # Add autonomous indicator to title if active
            if self.autonomous_indicator > 0:
                if self.autonomous_indicator % 6 < 3:  # Blink effect
                    title += " [AUTO]"
                self.autonomous_indicator -= 1

            # Status includes sleep cycle count if sleeping
            if self.sleep_state != 'awake':
                status = f"[Cycle: {self.sleep_cycle_count} | Frame: {self.animation_frame}]"
            else:
                status = f"[Frame: {self.animation_frame}]"

            title_x = max(2, (width - len(title)) // 2)
            status_x = width - len(status) - 3

            # Color based on sleep state
            if self.sleep_state == 'rem_sleep':
                title_color = self.color_pairs.get('high_weight', 0) or curses.A_BOLD
            elif self.sleep_state in ('deep_sleep', 'light_sleep'):
                title_color = self.color_pairs.get('user', 0) or curses.A_BOLD
            elif self.sleep_state == 'drowsy':
                title_color = self.color_pairs.get('metadata', 0) or curses.A_BOLD
            else:
                title_color = self.color_pairs.get('active', 0) or curses.A_BOLD

            self.stdscr.addstr(1, title_x, title, title_color)
            self.stdscr.addstr(1, status_x, status, self.color_pairs.get('metadata', 0))
        except curses.error:
            pass

    def _draw_memory_grid(self, height: int, width: int):
        """Draw the memory grid in Game of Life style"""
        grid_start_y = 3
        grid_end_y = height - 4
        grid_height = grid_end_y - grid_start_y

        if grid_height < 5:
            return

        try:
            # Draw zone labels
            zone_labels = [
                (width // 4, "THREAD"),
                (width // 2, "USER"),
                (3 * width // 4, "GLOBAL")
            ]

            for x, label in zone_labels:
                if 2 < x < width - 10:
                    self.stdscr.addstr(grid_start_y, x - len(label) // 2, label,
                                     self.color_pairs.get('metadata', 0))

            # Draw memory cells
            for (cell_x, cell_y), cell_data in self.memory_grid.items():
                # Map to screen coordinates
                screen_y = grid_start_y + 2 + (cell_y % (grid_height - 2))
                screen_x = 2 + (cell_x % (width - 4))

                if screen_y >= height - 3 or screen_x >= width - 2:
                    continue

                # Choose character based on weight and age
                weight = cell_data.get('weight', 0.5)
                age = cell_data.get('age', 0)
                scope = cell_data.get('scope', 'unknown')
                is_autonomous = cell_data.get('autonomous', False)

                # Special rendering for autonomous operations
                if is_autonomous:
                    # Autonomous operations use special characters
                    chars = ['[A]', '*', '@', '%', '*']
                    char = chars[self.animation_frame % len(chars)]
                    color = self.color_pairs.get('high_weight', 0) or curses.A_BOLD
                else:
                    # Character progression based on weight
                    if weight > 0.8:
                        char = '#'  # Solid block for high weight
                    elif weight > 0.6:
                        char = '='  # Dark shade
                    elif weight > 0.4:
                        char = '-'  # Medium shade
                    elif weight > 0.2:
                        char = '.'  # Light shade
                    else:
                        char = ','  # Dot for low weight

                    # Animate based on frame
                    if (cell_x, cell_y) in self.activity_cells:
                        if self.animation_frame % 4 < 2:
                            char = '*'  # Active indicator

                    # Get color for scope
                    color = self.color_pairs.get(scope, self.color_pairs.get('default', 0))

                    # Dim older memories
                    if age > 30:
                        color = self.color_pairs.get('metadata', 0)

                try:
                    self.stdscr.addstr(screen_y, screen_x, char, color)
                except curses.error:
                    pass

        except Exception:
            pass

    def _draw_stats_panel(self, height: int, width: int):
        """Draw statistics panel"""
        stats_y = height - 3

        try:
            # Build stats line with autonomous operation counts
            thread_count = self.memory_stats.get('thread', 0)
            user_count = self.memory_stats.get('user', 0)
            global_count = self.memory_stats.get('global', 0)
            total_count = self.memory_stats.get('total_recalls', 0)
            active_cells = len(self.activity_cells)

            rollup_count = self.autonomous_stats.get('rollups', 0)
            decay_count = self.autonomous_stats.get('decays', 0)
            consolidation_count = self.autonomous_stats.get('consolidations', 0)
            assessment_count = self.autonomous_stats.get('assessments', 0)

            # Add sleep discoveries if in sleep state
            if self.sleep_state != 'awake' and self.sleep_discoveries:
                patterns = self.sleep_discoveries.get('patterns', 0)
                relationships = self.sleep_discoveries.get('relationships', 0)
                insights = self.sleep_discoveries.get('insights', 0)

                stats_text = (f"T:{thread_count} U:{user_count} G:{global_count} "
                             f"| AUTO: R:{rollup_count} D:{decay_count} C:{consolidation_count} A:{assessment_count} "
                             f"| SLEEP: P:{patterns} REL:{relationships} I:{insights}")
            else:
                stats_text = (f"T:{thread_count} U:{user_count} G:{global_count} "
                             f"| Total:{total_count} Active:{active_cells} "
                             f"| AUTO: R:{rollup_count} D:{decay_count} C:{consolidation_count} A:{assessment_count}")

            if len(stats_text) < width - 4:
                self.stdscr.addstr(stats_y, 2, stats_text, self.color_pairs.get('metadata', 0))

        except curses.error:
            pass

    def _draw_recall_log(self, height: int, width: int):
        """Draw the rapid recall log at the bottom"""
        log_y = height - 2

        try:
            # Check if we should show autonomous operation or regular recall
            show_autonomous = False
            if self.autonomous_ops and self.recall_log:
                # Show autonomous if it's more recent
                latest_auto = self.autonomous_ops[-1]
                latest_recall = self.recall_log[-1]
                if latest_auto['time'] > latest_recall['time']:
                    show_autonomous = True
            elif self.autonomous_ops:
                show_autonomous = True

            if show_autonomous:
                # Show autonomous operation
                recent = self.autonomous_ops[-1]
                op_type = recent.get('type', 'unknown')
                scope = recent.get('scope', '')
                details = recent.get('details', '')
                count = recent.get('count', 0)

                timestamp = datetime.fromtimestamp(recent['time']).strftime('%H:%M:%S')
                log_text = f"[{timestamp}] [A] AUTONOMOUS: {op_type.upper()}"
                if scope:
                    log_text += f" scope={scope}"
                if count:
                    log_text += f" count={count}"
                if details:
                    log_text += f" | {details}"

                # Truncate to fit
                max_len = width - 4
                if len(log_text) > max_len:
                    log_text = log_text[:max_len - 3] + "..."

                color = self.color_pairs.get('high_weight', 0) or curses.A_BOLD
                self.stdscr.addstr(log_y, 2, log_text, color)

            elif self.recall_log:
                # Show regular recall
                recent = self.recall_log[-1]
                scope = recent.get('scope', '?')
                category = recent.get('category', '?')
                content = recent.get('content', '')
                weight = recent.get('weight', 0)

                # Format log line
                timestamp = datetime.fromtimestamp(recent['time']).strftime('%H:%M:%S')
                log_text = f"[{timestamp}] {scope}::{category} w={weight:.2f} | {content}"

                # Truncate to fit
                max_len = width - 4
                if len(log_text) > max_len:
                    log_text = log_text[:max_len - 3] + "..."

                # Choose color based on scope
                color = self.color_pairs.get(scope, self.color_pairs.get('default', 0))
                if weight > 0.7:
                    color = self.color_pairs.get('high_weight', color)

                self.stdscr.addstr(log_y, 2, log_text, color)
            else:
                self.stdscr.addstr(log_y, 2, "[Waiting for memory activity...]",
                                 self.color_pairs.get('metadata', 0))

        except curses.error:
            pass


# Global instance
_visualizer: Optional[MemoryVisualizer] = None


def get_visualizer(force_curses=False) -> MemoryVisualizer:
    """Get or create the global visualizer instance"""
    global _visualizer
    if _visualizer is None:
        _visualizer = MemoryVisualizer(force_curses=force_curses)
    return _visualizer


def start_visualizer(force_curses=False):
    """Start the memory visualizer

    Args:
        force_curses: If True, force curses mode even without a TTY (requires tmux/screen)
    """
    viz = get_visualizer(force_curses=force_curses)
    if not viz.running:
        viz.start()
    return viz


def stop_visualizer():
    """Stop the memory visualizer"""
    global _visualizer
    if _visualizer and _visualizer.running:
        _visualizer.stop()


# Convenience functions for logging
def log_recall(scope: str, category: str, content: str, weight: float, metadata: dict = None):
    """Log a memory recall"""
    viz = get_visualizer()
    if viz.running:
        viz.log_recall(scope, category, content, weight, metadata)


def log_operation(operation: str, **details):
    """Log a memory operation"""
    viz = get_visualizer()
    if viz.running:
        viz.log_memory_operation(operation, details)


def update_stats(**stats):
    """Update memory statistics"""
    viz = get_visualizer()
    if viz.running:
        viz.update_stats(stats)


def log_autonomous(op_type: str, scope: str = '', count: int = 0, details: str = ''):
    """Log an autonomous operation (rollup, decay, consolidation, assessment)"""
    viz = get_visualizer()
    if viz.running:
        try:
            viz.update_queue.put_nowait(('autonomous', {
                'type': op_type,
                'scope': scope,
                'count': count,
                'details': details,
                'time': time.time()
            }))
        except queue.Full:
            pass


def update_sleep_state(state: str, time_in_state: float = 0, cycle_count: int = 0, discoveries: dict = None):
    """Update the sleep cycle state in visualizer"""
    viz = get_visualizer()
    if viz.running:
        try:
            viz.update_queue.put_nowait(('sleep_state', {
                'state': state,
                'time_in_state': time_in_state,
                'cycle_count': cycle_count,
                'discoveries': discoveries or {}
            }))
        except queue.Full:
            pass


if __name__ == "__main__":
    # Demo mode
    print("Starting Memory Visualizer Demo...")
    print("Press 'q' in the visualizer to quit")
    print("\nThis demo simulates:")
    print("  - Memory recalls (thread, user, global)")
    print("  - Autonomous operations (rollup, decay, consolidation, assessment)")
    print("")

    viz = start_visualizer()
    time.sleep(1)

    # Simulate some memory recalls and autonomous operations
    try:
        scopes = ['thread', 'user', 'global']
        categories = ['thread_note', 'user_trait', 'global_theme', 'bias', 'relationship']
        auto_ops = ['rollup', 'decay', 'consolidation', 'assessment']

        for i in range(200):
            # Regular memory recall (70% of the time)
            if random.random() < 0.7:
                scope = random.choice(scopes)
                category = random.choice(categories)
                content = f"Memory recall {i}: {scope} memory about {category}"
                weight = random.random()

                log_recall(scope, category, content, weight)
                log_operation('recall', scope=scope, category=category)
            else:
                # Autonomous operation (30% of the time)
                op_type = random.choice(auto_ops)
                scope = random.choice(scopes)
                count = random.randint(1, 8)

                if op_type == 'rollup':
                    details = f'Hierarchical {scope} rollup'
                elif op_type == 'decay':
                    details = f'Memory limit exceeded for {scope}'
                elif op_type == 'consolidation':
                    details = f'Consolidated {count} memories'
                elif op_type == 'assessment':
                    details = f'Profile update for {scope}'
                else:
                    details = f'{op_type} operation'

                log_autonomous(op_type, scope=scope, count=count, details=details)

            time.sleep(0.15)

    except KeyboardInterrupt:
        pass
    finally:
        stop_visualizer()
        print("\nVisualizer stopped.")
