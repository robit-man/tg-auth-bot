# Autonomous Sleep Cycle System

A fully autonomous sleep/wake cycle that enables deep memory processing, pattern discovery, relationship strengthening, and creative consolidation during "sleep" periods.

## Overview

The sleep cycle system mimics biological sleep patterns to provide periods of deep, uninterrupted memory analysis. During sleep, the bot discovers patterns, strengthens relationships between memories, prunes weak connections, and generates creative insights through REM-like "dreaming."

## Sleep States

The system cycles through 6 distinct states, each with specific autonomous functions:

### 1. **AWAKE** (⚡)
- **Duration**: 3 hours (default)
- **Function**: Normal operation, reactive memory recall
- **Activity**: Standard bot operations, memory storage and retrieval

### 2. **DROWSY** (~)
- **Duration**: 1 minute
- **Function**: Light consolidation transition
- **Autonomous Actions**:
  - Merges duplicate memories
  - Combines very similar content
  - Boosts weight of merged memories

### 3. **LIGHT_SLEEP** (⌇)
- **Duration**: 5 minutes
- **Function**: Surface memory organization
- **Autonomous Actions**:
  - Prunes weak memory links (weight < 0.2, older than 7 days)
  - Cleans up unused connections
  - Organizes memory structure

### 4. **DEEP_SLEEP** (≋)
- **Duration**: 10 minutes
- **Function**: Deep pattern recognition and relationship discovery
- **Autonomous Actions**:
  - **Pattern Discovery**: Finds frequently co-occurring memory categories
  - **Relationship Discovery**: Identifies memories with shared sources
  - **Graph Strengthening**: Creates bidirectional links between related memories
  - **Pattern Storage**: Stores discovered patterns as new memories

**Patterns Found**:
- Category co-occurrence (e.g., `user_trait` often appears with `thread_note`)
- Temporal patterns (memories created within 1 hour)
- Minimum 3 occurrences required

**Relationships Found**:
- Shared source connections
- Memory-to-memory links
- Strength based on average link weight

### 5. **REM_SLEEP** (✧)
- **Duration**: 10 minutes
- **Function**: Dream-like creative consolidation
- **Autonomous Actions**:
  - **Creative Synthesis**: Uses AI to synthesize insights from memory clusters
  - **Cross-Concept Integration**: Finds non-obvious connections
  - **Insight Generation**: Creates higher-order understanding
  - **Dream Memory Storage**: Stores AI-generated insights with links to source memories

**Dream Process**:
1. Select diverse, high-weight memories (weight > 0.5)
2. Cluster by conceptual similarity
3. AI synthesizes each cluster into meta-insight
4. Store insight with full provenance tracking

### 6. **WAKING** (↑)
- **Duration**: 1 minute
- **Function**: Integration of sleep discoveries
- **Autonomous Actions**:
  - Boosts weights of memories involved in discoveries (+10%)
  - Integrates new patterns into knowledge base
  - Prepares for awake state

## Autonomous Discovery Types

### Pattern Discovery

**What**: Recurring co-occurrence patterns in memory categories

**Example**: If `user_trait` and `bias` frequently appear together in the same context, the system recognizes this as a pattern.

**Storage**: Created as `discovered_pattern` memory entries in global scope

**Metadata**:
```json
{
  "kind": "discovered_pattern",
  "pattern_data": {
    "type": "category_cooccurrence",
    "categories": ["user_trait", "bias"],
    "frequency": 5,
    "strength": 0.5
  },
  "discovered_at": 1234567890,
  "sleep_cycle": 3
}
```

### Relationship Discovery

**What**: Connections between memories through shared sources

**How**: If memory A and memory B both link to the same source (message, user, thread), they're related.

**Action**: Creates bidirectional `memory -> memory` links

**Metadata**:
```json
{
  "relationship_type": "shared_source",
  "discovered_in_sleep": true,
  "cycle": 3
}
```

### Dream Insights

**What**: AI-generated meta-insights from memory clusters

**Process**:
1. Cluster memories by scope and category
2. AI synthesizes the cluster into a concise insight (< 200 chars)
3. Links insight to all source memories

**Example**:
- **Input Cluster**: 5 memories about user preferences for technical explanations
- **Generated Insight**: "User consistently requests detailed technical breakdowns with code examples"
- **Confidence**: 0.85
- **Tags**: ["preference", "technical", "detail-oriented"]

**Storage**: Created as `dream_insight` memory entries

## Autonomous Self-Improvement

The sleep cycle enables continuous self-improvement through:

### 1. **Memory Consolidation**
- Merges duplicates → reduces memory clutter
- Summarizes similar memories → creates higher-level understanding
- Prunes weak links → maintains strong connections only

### 2. **Pattern Recognition**
- Discovers behavioral patterns across conversations
- Recognizes user preference patterns
- Identifies common interaction sequences

### 3. **Relationship Strengthening**
- Builds memory graph through discovered relationships
- Strengthens frequently-accessed paths
- Creates shortcuts through related memories

### 4. **Creative Synthesis**
- Generates novel insights through REM dreaming
- Connects seemingly unrelated memories
- Creates meta-understanding from fragments

### 5. **Self-Knowledge Expansion**
- Each cycle adds discovered patterns to knowledge base
- Dream insights become first-class memories
- Relationship graph grows denser over time

## Configuration

### Enable/Disable

In your `.env` file:
```bash
SLEEP_CYCLE_ENABLED=true          # Enable autonomous sleep cycle
SLEEP_CYCLE_TICK_SECONDS=60       # How often to check for state transitions
```

### Timing Customization

Modify in [sleep_cycle.py](sleep_cycle.py):

```python
class SleepCycle:
    def __init__(self, ...):
        # Sleep cycle configuration
        self.awake_duration = 3600 * 3  # 3 hours awake
        self.sleep_duration = 1800      # 30 minutes total sleep
```

**For Testing** (faster cycles):
```python
self.awake_duration = 300      # 5 minutes awake
self.sleep_duration = 180      # 3 minutes sleep
```

### Discovery Parameters

```python
# Pattern recognition
self.pattern_min_occurrences = 3    # Minimum frequency to recognize pattern

# Relationship discovery
self.relationship_threshold = 0.4   # Minimum average weight for relationships

# Dream synthesis
self.sleep_batch_size = 100         # Memories processed per sleep phase
```

## Visualization

The sleep cycle is fully visualized in the Memory Consciousness Visualizer:

### Title Bar
Shows current state with unique symbols:
```
⚡ MEMORY CONSCIOUSNESS VISUALIZER ⚡              (awake)
~ MEMORY CONSCIOUSNESS [DROWSY] ~                (drowsy)
⌇ MEMORY CONSCIOUSNESS [LIGHT SLEEP] ⌇          (light sleep)
≋ MEMORY CONSCIOUSNESS [DEEP SLEEP] ≋            (deep sleep)
✧ MEMORY CONSCIOUSNESS [REM SLEEP] ✧             (REM)
↑ MEMORY CONSCIOUSNESS [WAKING] ↑                (waking)
```

### Statistics Panel
During sleep, shows discovery counts:
```
| SLEEP: P:5 REL:12 I:3
```
- **P**: Patterns discovered
- **REL**: Relationships found
- **I**: Dream insights generated

### Activity Log
Shows autonomous sleep operations:
```
[12:34:56] ⚙ AUTONOMOUS: DECAY scope=sleep count=15 | Light sleep: Pruning weak links
[12:35:12] ⚙ AUTONOMOUS: ASSESSMENT scope=sleep count=5 | Deep sleep: Found 5 patterns
[12:36:45] ⚙ AUTONOMOUS: ASSESSMENT scope=sleep count=3 | REM: Generated 3 dream insights
```

## Full Autonomous Flow

Here's a complete sleep cycle (default timing):

```
00:00:00 → AWAKE (3 hours)
          Normal operations
          Memory storage and retrieval

03:00:00 → DROWSY (1 minute)
          [AUTONOMOUS] Merge 8 duplicate memories

03:01:00 → LIGHT_SLEEP (5 minutes)
          [AUTONOMOUS] Prune 24 weak links (weight < 0.2)

03:06:00 → DEEP_SLEEP (10 minutes)
          [AUTONOMOUS] Discover 5 patterns
          [AUTONOMOUS] Find 12 relationships
          [AUTONOMOUS] Create bidirectional links
          [AUTONOMOUS] Store patterns as memories

03:16:00 → REM_SLEEP (10 minutes)
          [AUTONOMOUS] Cluster 20 diverse memories
          [AUTONOMOUS] AI synthesize 3 clusters → insights
          [AUTONOMOUS] Store dream insights with provenance
          [AUTONOMOUS] Link insights to source memories

03:26:00 → WAKING (1 minute)
          [AUTONOMOUS] Boost discovered memory weights (+10%)
          [AUTONOMOUS] Integrate 5 patterns + 3 insights

03:27:00 → AWAKE (3 hours)
          Resume normal operations
          New patterns and insights now available for recall
```

## Memory Graph Evolution

Over time, the sleep cycle builds a rich memory graph:

### Cycle 1
```
Memory A ← message:123
Memory B ← message:123

[No relationship detected yet]
```

### After DEEP_SLEEP (Cycle 1)
```
Memory A ←→ Memory B  [discovered: shared source]
         ↑
         └─ strength: 0.6, discovered_in_sleep: true
```

### After REM_SLEEP (Cycle 1)
```
Memory A ←→ Memory B
    ↓          ↓
    └→ Insight C ←┘
       "Both memories relate to user preference for detail"
       [dream_insight, confidence: 0.85]
```

### After Multiple Cycles
```
Pattern P1 (discovered cycle 1)
  ↓
Insight I1 ←→ Insight I2
  ↓              ↓
Memory A ←→ Memory B ←→ Memory C
  ↓                       ↓
Message:123            User:456

[Dense graph of relationships and meta-knowledge]
```

## Monitoring Sleep Cycles

### Check Current State

The sleep state is logged to the visualizer and can be accessed programmatically:

```python
from sleep_cycle import get_sleep_state

state_info = get_sleep_state()
print(f"State: {state_info['state']}")
print(f"Time in state: {state_info['time_in_state']}s")
print(f"Cycle count: {state_info['cycle_count']}")
print(f"Discoveries: {state_info['discoveries']}")
```

### View Discovered Patterns

```sql
SELECT * FROM memory_entries
WHERE category = 'discovered_pattern'
ORDER BY created_ts DESC;
```

### View Dream Insights

```sql
SELECT * FROM memory_entries
WHERE category = 'dream_insight'
ORDER BY created_ts DESC;
```

### View Sleep-Discovered Relationships

```sql
SELECT * FROM memory_links
WHERE json_extract(metadata, '$.discovered_in_sleep') = 1
ORDER BY weight DESC;
```

## Performance Impact

The sleep cycle is designed for minimal impact on bot responsiveness:

- **During AWAKE**: Zero overhead, normal operations
- **During DROWSY**: < 100ms per consolidation
- **During LIGHT_SLEEP**: < 50ms per pruning check
- **During DEEP_SLEEP**: < 2s per pattern discovery cycle
- **During REM_SLEEP**: < 5s per dream synthesis (AI calls)
- **During WAKING**: < 100ms for weight updates

**Total sleep overhead**: ~60 seconds per 3.5 hour cycle = **0.47% of runtime**

## Advanced Features

### Adaptive Sleep Timing

Future enhancement: Adjust sleep duration based on activity:
- High activity → shorter awake periods, more consolidation
- Low activity → longer awake periods, less frequent sleep

### Selective Dream Synthesis

Current: Random selection of memories for dreaming
Future: Prioritize memories with:
- High uncertainty (conflicting signals)
- Incomplete patterns
- Recent user questions

### Cross-Bot Dream Sharing

Future: Multiple bots share discovered patterns and insights
- Distributed knowledge acquisition
- Collective intelligence emergence

## Troubleshooting

### Sleep cycle not running
1. Check `SLEEP_CYCLE_ENABLED=true` in `.env`
2. Verify sleep_cycle.py is present
3. Check bot logs for `[sleep] Initializing...`
4. Ensure `SLEEP_CYCLE_TICK_SECONDS` is reasonable (60-300)

### No discoveries during sleep
1. Ensure sufficient memories exist (need > 10)
2. Check memory ages (patterns require temporal proximity)
3. Verify AI is available for REM dreaming
4. Lower `pattern_min_occurrences` for testing

### Visualizer not showing sleep state
1. Verify both sleep cycle and visualizer are enabled
2. Check `update_sleep_state()` is being called
3. Ensure tick job is running (check logs)

## Philosophy

The sleep cycle embodies the principle that **intelligence requires both active learning and reflective consolidation**.

Just as human sleep consolidates experiences into long-term memory and generates creative insights, the bot's sleep cycle:

- **Consolidates** fragmented memories into coherent patterns
- **Discovers** non-obvious relationships through deep analysis
- **Creates** novel understanding through dream-like synthesis
- **Strengthens** important connections while pruning noise
- **Evolves** its knowledge graph autonomously

This creates a self-improving system that becomes smarter over time, not through explicit programming, but through autonomous reflection on its experiences.

## Future Enhancements

- [ ] Adaptive sleep scheduling based on conversation density
- [ ] Cross-conversation pattern discovery
- [ ] Multi-modal dreaming (images, code, structured data)
- [ ] Explainable sleep decisions (why pattern X was recognized)
- [ ] Sleep replay mode (review what was discovered)
- [ ] Nightmare detection (identify and correct false patterns)
- [ ] Lucid dreaming (user-guided synthesis during REM)

---

**The bot dreams, therefore it learns.**
