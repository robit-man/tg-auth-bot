#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sleep Cycle System - Autonomous deep memory analysis and consolidation

During "sleep" periods, the bot performs deep analysis of its memory stream,
discovers relationships, strengthens important connections, and improves
its understanding of stored information through dream-like consolidation.
"""

import time
import json
import math
import random
import sqlite3
import textwrap
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone

# Sleep cycle states
class SleepState:
    AWAKE = "awake"
    DROWSY = "drowsy"
    LIGHT_SLEEP = "light_sleep"
    DEEP_SLEEP = "deep_sleep"
    REM_SLEEP = "rem_sleep"  # Dream state - deep consolidation
    WAKING = "waking"


class SleepCycle:
    """
    Manages autonomous sleep/wake cycles for deep memory processing

    Sleep Phases:
    1. AWAKE: Normal operation, reactive memory recall
    2. DROWSY: Transition state, light consolidation begins
    3. LIGHT_SLEEP: Surface memory organization, weak link pruning
    4. DEEP_SLEEP: Deep pattern recognition, relationship discovery
    5. REM_SLEEP: Dream-like consolidation, creative connections
    6. WAKING: Integration of sleep insights, cache warming
    """

    def __init__(self, db_path: str, visualizer_log_fn=None):
        self.db_path = db_path
        self.log_autonomous = visualizer_log_fn or (lambda *a, **k: None)

        # Sleep cycle configuration
        self.awake_duration = 3600 * 3  # 3 hours awake
        self.sleep_duration = 1800      # 30 minutes sleep

        # Current state
        self.current_state = SleepState.AWAKE
        self.state_entered_at = time.time()
        self.sleep_cycle_count = 0
        self.last_deep_analysis = 0

        # Sleep metrics
        self.discoveries = {
            'relationships': 0,
            'patterns': 0,
            'consolidations': 0,
            'prunings': 0,
            'insights': 0,
        }

        # Memory working set for sleep processing
        self.sleep_batch_size = 100
        self.relationship_threshold = 0.4
        self.pattern_min_occurrences = 3

    def get_state(self) -> Dict:
        """Get current sleep state information"""
        time_in_state = time.time() - self.state_entered_at
        return {
            'state': self.current_state,
            'time_in_state': time_in_state,
            'cycle_count': self.sleep_cycle_count,
            'discoveries': self.discoveries.copy(),
        }

    def should_transition(self) -> bool:
        """Check if we should transition to next sleep state"""
        elapsed = time.time() - self.state_entered_at

        if self.current_state == SleepState.AWAKE:
            return elapsed >= self.awake_duration
        elif self.current_state == SleepState.DROWSY:
            return elapsed >= 60  # 1 minute drowsy
        elif self.current_state == SleepState.LIGHT_SLEEP:
            return elapsed >= 300  # 5 minutes light sleep
        elif self.current_state == SleepState.DEEP_SLEEP:
            return elapsed >= 600  # 10 minutes deep sleep
        elif self.current_state == SleepState.REM_SLEEP:
            return elapsed >= 600  # 10 minutes REM
        elif self.current_state == SleepState.WAKING:
            return elapsed >= 60  # 1 minute waking

        return False

    def transition_state(self) -> str:
        """Transition to next sleep state"""
        transitions = {
            SleepState.AWAKE: SleepState.DROWSY,
            SleepState.DROWSY: SleepState.LIGHT_SLEEP,
            SleepState.LIGHT_SLEEP: SleepState.DEEP_SLEEP,
            SleepState.DEEP_SLEEP: SleepState.REM_SLEEP,
            SleepState.REM_SLEEP: SleepState.WAKING,
            SleepState.WAKING: SleepState.AWAKE,
        }

        old_state = self.current_state
        self.current_state = transitions.get(old_state, SleepState.AWAKE)
        self.state_entered_at = time.time()

        if self.current_state == SleepState.DROWSY:
            self.sleep_cycle_count += 1

        self.log_autonomous(
            'assessment',
            scope='sleep_cycle',
            count=self.sleep_cycle_count,
            details=f'State transition: {old_state} -> {self.current_state}'
        )

        return self.current_state

    async def process_sleep_phase(self, ai_generate_fn=None):
        """Process current sleep phase"""
        if self.current_state == SleepState.AWAKE:
            # Normal operation - no sleep processing
            pass

        elif self.current_state == SleepState.DROWSY:
            # Light consolidation - merge very similar memories
            await self._drowsy_consolidation()

        elif self.current_state == SleepState.LIGHT_SLEEP:
            # Prune weak links and organize surface memories
            await self._light_sleep_organization()

        elif self.current_state == SleepState.DEEP_SLEEP:
            # Deep pattern recognition and relationship discovery
            await self._deep_sleep_analysis(ai_generate_fn)

        elif self.current_state == SleepState.REM_SLEEP:
            # Dream-like creative consolidation
            await self._rem_sleep_dreaming(ai_generate_fn)

        elif self.current_state == SleepState.WAKING:
            # Integration and cache warming
            await self._waking_integration()

    async def _drowsy_consolidation(self):
        """Light consolidation during drowsy phase"""
        self.log_autonomous('consolidation', scope='sleep', details='Drowsy: Light consolidation')

        # Find very similar memories (high duplicate potential)
        with closing(sqlite3.connect(self.db_path)) as con:
            memories = con.execute("""
                SELECT id, scope, chat_id, thread_id, user_id, category, content, weight, created_ts
                FROM memory_entries
                ORDER BY created_ts DESC
                LIMIT ?
            """, (self.sleep_batch_size,)).fetchall()

        # Group by scope and category
        groups = defaultdict(list)
        for mem in memories:
            key = (mem[1], mem[2], mem[3], mem[4], mem[5])  # scope, chat, thread, user, category
            groups[key].append(mem)

        consolidated = 0
        for key, group in groups.items():
            if len(group) < 2:
                continue

            # Find duplicates by content similarity (simple string matching for now)
            seen = {}
            to_merge = []

            for mem in group:
                content_key = mem[6][:100].lower().strip()  # First 100 chars
                if content_key in seen:
                    to_merge.append((seen[content_key], mem))
                else:
                    seen[content_key] = mem

            # Merge duplicates
            for primary, duplicate in to_merge:
                await self._merge_memories(primary, duplicate)
                consolidated += 1

        if consolidated > 0:
            self.discoveries['consolidations'] += consolidated
            self.log_autonomous('consolidation', scope='sleep', count=consolidated,
                              details=f'Drowsy: Merged {consolidated} duplicate memories')

    async def _light_sleep_organization(self):
        """Organize and prune during light sleep"""
        self.log_autonomous('decay', scope='sleep', details='Light sleep: Pruning weak links')

        # Prune weak memory links
        with closing(sqlite3.connect(self.db_path)) as con:
            # Find weak links (low weight, old, unused)
            weak_links = con.execute("""
                SELECT memory_id, source_type, source_id, weight
                FROM memory_links
                WHERE weight < 0.2
                AND memory_id IN (
                    SELECT id FROM memory_entries
                    WHERE created_ts < ?
                )
            """, (int(time.time()) - 86400 * 7,)).fetchall()  # Older than 7 days

            if weak_links:
                # Delete weak links
                for memory_id, source_type, source_id, weight in weak_links:
                    con.execute("""
                        DELETE FROM memory_links
                        WHERE memory_id=? AND source_type=? AND source_id=?
                    """, (memory_id, source_type, source_id))

                con.commit()

                self.discoveries['prunings'] += len(weak_links)
                self.log_autonomous('decay', scope='sleep', count=len(weak_links),
                                  details=f'Light sleep: Pruned {len(weak_links)} weak links')

    async def _deep_sleep_analysis(self, ai_generate_fn=None):
        """Deep pattern recognition and relationship discovery"""
        self.log_autonomous('assessment', scope='sleep', details='Deep sleep: Pattern recognition')

        # Analyze memory patterns
        patterns = await self._discover_patterns()
        relationships = await self._discover_relationships()

        self.discoveries['patterns'] += len(patterns)
        self.discoveries['relationships'] += len(relationships)

        if patterns:
            self.log_autonomous('assessment', scope='sleep', count=len(patterns),
                              details=f'Deep sleep: Found {len(patterns)} patterns')

        if relationships:
            self.log_autonomous('assessment', scope='sleep', count=len(relationships),
                              details=f'Deep sleep: Found {len(relationships)} relationships')

        # Store discovered patterns as new memories
        for pattern in patterns[:5]:  # Limit to top 5
            await self._store_pattern_memory(pattern)

        # Strengthen relationships
        for rel in relationships[:10]:  # Limit to top 10
            await self._strengthen_relationship(rel)

    async def _rem_sleep_dreaming(self, ai_generate_fn=None):
        """Dream-like creative consolidation during REM sleep"""
        self.log_autonomous('assessment', scope='sleep', details='REM: Dream consolidation')

        if not ai_generate_fn:
            return

        # Select diverse memories for creative synthesis
        with closing(sqlite3.connect(self.db_path)) as con:
            memories = con.execute("""
                SELECT id, scope, category, content, weight, metadata
                FROM memory_entries
                WHERE weight > 0.5
                ORDER BY RANDOM()
                LIMIT 20
            """).fetchall()

        if len(memories) < 3:
            return

        # Group into conceptual clusters
        clusters = self._cluster_memories(memories)

        insights = 0
        for cluster in clusters[:3]:  # Process top 3 clusters
            insight = await self._dream_synthesis(cluster, ai_generate_fn)
            if insight:
                await self._store_dream_insight(insight)
                insights += 1

        if insights > 0:
            self.discoveries['insights'] += insights
            self.log_autonomous('assessment', scope='sleep', count=insights,
                              details=f'REM: Generated {insights} dream insights')

    async def _waking_integration(self):
        """Integration phase while waking"""
        self.log_autonomous('assessment', scope='sleep', details='Waking: Integrating sleep discoveries')

        # Update memory weights based on sleep discoveries
        with closing(sqlite3.connect(self.db_path)) as con:
            # Boost weights of memories that were involved in discoveries
            con.execute("""
                UPDATE memory_entries
                SET weight = MIN(1.0, weight * 1.1)
                WHERE updated_ts > ?
            """, (int(self.state_entered_at),))

            con.commit()

        summary = ', '.join([f'{k}:{v}' for k, v in self.discoveries.items() if v > 0])
        self.log_autonomous('assessment', scope='sleep', count=sum(self.discoveries.values()),
                          details=f'Waking: Discoveries - {summary}')

    async def _discover_patterns(self) -> List[Dict]:
        """Discover recurring patterns in memories"""
        with closing(sqlite3.connect(self.db_path)) as con:
            # Find frequently co-occurring categories
            category_pairs = con.execute("""
                SELECT m1.category, m2.category, COUNT(*) as cnt
                FROM memory_entries m1
                JOIN memory_entries m2 ON
                    m1.scope = m2.scope AND
                    m1.chat_id = m2.chat_id AND
                    m1.thread_id = m2.thread_id AND
                    m1.id < m2.id AND
                    ABS(m1.created_ts - m2.created_ts) < 3600
                WHERE m1.category != m2.category
                GROUP BY m1.category, m2.category
                HAVING cnt >= ?
                ORDER BY cnt DESC
                LIMIT 10
            """, (self.pattern_min_occurrences,)).fetchall()

        patterns = []
        for cat1, cat2, count in category_pairs:
            patterns.append({
                'type': 'category_cooccurrence',
                'categories': [cat1, cat2],
                'frequency': count,
                'strength': min(1.0, count / 10.0)
            })

        return patterns

    async def _discover_relationships(self) -> List[Dict]:
        """Discover relationships between memories"""
        relationships = []

        with closing(sqlite3.connect(self.db_path)) as con:
            # Find memories with shared sources
            shared_sources = con.execute("""
                SELECT l1.memory_id, l2.memory_id, l1.source_type, l1.source_id,
                       AVG(l1.weight + l2.weight) / 2 as avg_weight
                FROM memory_links l1
                JOIN memory_links l2 ON
                    l1.source_type = l2.source_type AND
                    l1.source_id = l2.source_id AND
                    l1.memory_id < l2.memory_id
                GROUP BY l1.memory_id, l2.memory_id, l1.source_type, l1.source_id
                HAVING avg_weight > ?
                LIMIT 50
            """, (self.relationship_threshold,)).fetchall()

        for mem1_id, mem2_id, source_type, source_id, avg_weight in shared_sources:
            relationships.append({
                'memory_ids': [mem1_id, mem2_id],
                'source_type': source_type,
                'source_id': source_id,
                'strength': avg_weight,
                'type': 'shared_source'
            })

        return relationships

    def _cluster_memories(self, memories: List[Tuple]) -> List[List[Dict]]:
        """Cluster memories by conceptual similarity"""
        # Simple clustering by category and scope
        clusters = defaultdict(list)

        for mem in memories:
            mem_id, scope, category, content, weight, metadata_json = mem
            key = f"{scope}_{category}"

            try:
                metadata = json.loads(metadata_json or '{}')
            except:
                metadata = {}

            clusters[key].append({
                'id': mem_id,
                'scope': scope,
                'category': category,
                'content': content,
                'weight': weight,
                'metadata': metadata
            })

        # Return clusters with at least 2 memories
        return [cluster for cluster in clusters.values() if len(cluster) >= 2]

    async def _dream_synthesis(self, cluster: List[Dict], ai_generate_fn) -> Optional[Dict]:
        """Synthesize insight from memory cluster during REM sleep"""
        if not cluster or not ai_generate_fn:
            return None

        # Prepare cluster contents for AI
        contents = [m['content'] for m in cluster[:5]]
        scope = cluster[0]['scope']
        category = cluster[0]['category']

        prompt = textwrap.dedent(f"""
            During deep sleep consolidation, synthesize these related memories into a higher-order insight.
            Respond with JSON: {{"insight": "concise synthesis", "confidence": 0.0-1.0, "tags": ["tag1", "tag2"]}}

            Memory cluster ({scope}::{category}):
            {json.dumps(contents, ensure_ascii=False, indent=2)}

            Create a meta-insight that captures the essence or pattern across these memories.
            Keep it under 200 characters.
        """).strip()

        try:
            result = await ai_generate_fn({
                "model": "llama3.2",  # Placeholder - should use actual model
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            })

            if not result:
                return None

            # Parse JSON response
            cleaned = result.strip()
            if '```json' in cleaned:
                cleaned = cleaned.split('```json')[1].split('```')[0]
            elif '```' in cleaned:
                cleaned = cleaned.split('```')[1].split('```')[0]

            data = json.loads(cleaned)

            return {
                'insight': data.get('insight', ''),
                'confidence': float(data.get('confidence', 0.5)),
                'tags': data.get('tags', []),
                'source_memories': [m['id'] for m in cluster],
                'scope': scope,
                'category': category
            }

        except Exception:
            return None

    async def _merge_memories(self, primary: Tuple, duplicate: Tuple):
        """Merge duplicate memory into primary"""
        primary_id = primary[0]
        duplicate_id = duplicate[0]

        with closing(sqlite3.connect(self.db_path)) as con:
            # Transfer links from duplicate to primary
            con.execute("""
                INSERT OR IGNORE INTO memory_links (memory_id, source_type, source_id, weight, metadata)
                SELECT ?, source_type, source_id, weight, metadata
                FROM memory_links
                WHERE memory_id = ?
            """, (primary_id, duplicate_id))

            # Boost primary weight slightly
            new_weight = min(1.0, (primary[7] + duplicate[7]) / 2 + 0.1)
            con.execute("UPDATE memory_entries SET weight=? WHERE id=?", (new_weight, primary_id))

            # Delete duplicate
            con.execute("DELETE FROM memory_entries WHERE id=?", (duplicate_id,))
            con.execute("DELETE FROM memory_links WHERE memory_id=?", (duplicate_id,))

            con.commit()

    async def _store_pattern_memory(self, pattern: Dict):
        """Store discovered pattern as a new memory"""
        with closing(sqlite3.connect(self.db_path)) as con:
            content = f"Pattern: {pattern['type']} - {', '.join(pattern.get('categories', []))}"
            metadata = json.dumps({
                'kind': 'discovered_pattern',
                'pattern_data': pattern,
                'discovered_at': int(time.time()),
                'sleep_cycle': self.sleep_cycle_count
            }, ensure_ascii=False)

            con.execute("""
                INSERT INTO memory_entries
                (scope, chat_id, thread_id, user_id, category, content, metadata, weight, created_ts, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ('global', 0, 0, 0, 'discovered_pattern', content, metadata,
                  pattern.get('strength', 0.6), int(time.time()), int(time.time())))

            con.commit()

    async def _strengthen_relationship(self, relationship: Dict):
        """Strengthen a discovered relationship between memories"""
        mem1_id, mem2_id = relationship['memory_ids']
        strength = relationship['strength']

        with closing(sqlite3.connect(self.db_path)) as con:
            # Create bidirectional link if not exists
            con.execute("""
                INSERT OR REPLACE INTO memory_links (memory_id, source_type, source_id, weight, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (mem1_id, 'memory', mem2_id, strength, json.dumps({
                'relationship_type': relationship['type'],
                'discovered_in_sleep': True,
                'cycle': self.sleep_cycle_count
            })))

            con.execute("""
                INSERT OR REPLACE INTO memory_links (memory_id, source_type, source_id, weight, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (mem2_id, 'memory', mem1_id, strength, json.dumps({
                'relationship_type': relationship['type'],
                'discovered_in_sleep': True,
                'cycle': self.sleep_cycle_count
            })))

            con.commit()

    async def _store_dream_insight(self, insight: Dict):
        """Store a dream-generated insight as memory"""
        if not insight or not insight.get('insight'):
            return

        with closing(sqlite3.connect(self.db_path)) as con:
            metadata = json.dumps({
                'kind': 'dream_insight',
                'confidence': insight['confidence'],
                'tags': insight['tags'],
                'source_memories': insight['source_memories'],
                'sleep_cycle': self.sleep_cycle_count,
                'generated_at': int(time.time())
            }, ensure_ascii=False)

            # Create links to source memories
            con.execute("""
                INSERT INTO memory_entries
                (scope, chat_id, thread_id, user_id, category, content, metadata, weight, created_ts, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (insight['scope'], 0, 0, 0, 'dream_insight', insight['insight'],
                  metadata, insight['confidence'], int(time.time()), int(time.time())))

            dream_memory_id = con.lastrowid

            # Link to source memories
            for source_mem_id in insight['source_memories']:
                con.execute("""
                    INSERT OR IGNORE INTO memory_links (memory_id, source_type, source_id, weight, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (dream_memory_id, 'memory', source_mem_id, 0.8, json.dumps({'dream_source': True})))

            con.commit()


# Global sleep cycle instance
_sleep_cycle: Optional[SleepCycle] = None


def init_sleep_cycle(db_path: str, visualizer_log_fn=None) -> SleepCycle:
    """Initialize the global sleep cycle"""
    global _sleep_cycle
    _sleep_cycle = SleepCycle(db_path, visualizer_log_fn)
    return _sleep_cycle


def get_sleep_cycle() -> Optional[SleepCycle]:
    """Get the global sleep cycle instance"""
    return _sleep_cycle


async def sleep_cycle_tick(ai_generate_fn=None):
    """Called periodically to advance sleep cycle"""
    if not _sleep_cycle:
        return

    # Check for state transition
    if _sleep_cycle.should_transition():
        _sleep_cycle.transition_state()

    # Process current sleep phase
    await _sleep_cycle.process_sleep_phase(ai_generate_fn)


def is_sleeping() -> bool:
    """Check if currently in sleep state"""
    if not _sleep_cycle:
        return False

    return _sleep_cycle.current_state not in [SleepState.AWAKE, SleepState.WAKING]


def get_sleep_state() -> Dict:
    """Get current sleep state info"""
    if not _sleep_cycle:
        return {
            'state': SleepState.AWAKE,
            'time_in_state': 0,
            'cycle_count': 0,
            'discoveries': {}
        }

    return _sleep_cycle.get_state()
