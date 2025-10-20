# Intelligent Agent System - Complete Implementation

## Executive Summary

The telegram bot now features an intelligent agent system based on the `/examples/` reference implementation, with automatic tool selection, document RAG, and vision capabilities.

**Status**: ✅ **COMPLETE** - All components integrated and ready to test

## What Was Implemented

### 1. Intelligent Tool Router (`tool_router.py` - 462 lines)

**Purpose**: Automatically analyzes user intent and routes to appropriate tools/capabilities.

**Key Features**:
- LLM-based route selection using Ollama
- Heuristic keyword matching for fast routing
- Score aggregation (LLM + heuristics)
- Confidence scoring for decision quality
- Automatic tool parameter extraction
- Support for multiple route types

**Routes Available**:
```python
RouteType.DIRECT_ANSWER     # Simple conversation
RouteType.WEB_SEARCH        # Internet search
RouteType.KNOWLEDGE_QUERY   # RAG over documents
RouteType.FILE_OPERATION    # File system ops
RouteType.IMAGE_ANALYSIS    # Vision analysis
RouteType.TOOL_EXECUTION    # Explicit tool calls
```

**How It Works**:
1. User sends message
2. Router calculates heuristic scores based on keywords
3. If LLM routing enabled, gets LLM route scores
4. Aggregates scores (heuristics + LLM * 2.0)
5. Selects best route with confidence score
6. Determines which specific tools are needed
7. Executes tools and returns results

**Example**:
```python
from tool_router import IntelligentToolRouter

router = IntelligentToolRouter(
    ollama_model="llama3.2",
    enable_llm_routing=True,
)

# Analyze user intent
decision = router.analyze_intent(
    user_message="Search for Python 3.12 features",
    context={"user_id": 123}
)

# decision.route == RouteType.WEB_SEARCH
# decision.confidence == 0.85
# decision.tools_needed == ["search_internet"]

# Execute the route
response, results = await router.execute_route(decision, user_message)
```

### 2. Document RAG System (`document_rag.py` - 701 lines)

**Purpose**: Hybrid retrieval-augmented generation over uploaded documents.

**Key Features**:
- Multi-format parsing (PDF, DOCX, TXT, HTML, MD)
- PDF page vision extraction with Ollama multimodal models
- Vector embeddings for semantic search
- FTS5 keyword search (when available)
- Reciprocal Rank Fusion (RRF) for hybrid retrieval
- SQLite storage with provenance tracking
- Chunk-based retrieval with metadata

**Supported Formats**:
- PDF (with optional vision for tables/figures)
- DOCX (Microsoft Word)
- TXT (with charset detection)
- HTML/HTM
- Markdown (.md, .markdown)

**How It Works**:
1. User uploads document via Telegram
2. File downloaded to `data/uploads/`
3. Document parsed and split into chunks
4. Chunks embedded using Ollama embed model
5. Optional: PDF pages rendered to PNG and vision-extracted
6. Stored in SQLite with vectors
7. Query uses vector similarity + keyword search
8. Results merged with RRF

**Database Schema**:
```sql
documents(
    id, filename, file_path, checksum, file_type,
    user_id, chat_id, message_id, added_ts, status
)

chunks(
    id, doc_id, chunk_index, text, page_no,
    modality, resource_path, ts
)

chunk_vectors(
    chunk_id, embedding BLOB, dim, norm
)

chunks_fts(
    chunk_id, content  -- FTS5 virtual table
)
```

**Example**:
```python
from document_rag import DocumentRAGStore
from pathlib import Path

rag = DocumentRAGStore(
    db_path=Path("data/documents.db"),
    embed_model="nomic-embed-text",
    vision_model="llava",
)

# Ingest document
doc_id = rag.add_document_from_telegram(
    file_path=Path("example.pdf"),
    user_id=123,
    chat_id=456,
    message_id=789,
    use_vision=True,  # Extract text from images
)

# Search
result = rag.search(
    query="What are the key findings?",
    top_k=5,
    user_id=123,
)

for chunk in result.chunks:
    print(f"Page {chunk.page_no}: {chunk.text}")
```

### 3. Vision Handler (`vision_handler.py` - 306 lines)

**Purpose**: Image analysis using Ollama multimodal models (llava, bakllava, etc.).

**Key Features**:
- Single image analysis
- Context-aware image analysis (with conversation history)
- Multiple image analysis
- Image comparison
- Base64 encoding for Ollama API
- Streaming support for responses

**How It Works**:
1. User uploads photo via Telegram
2. Image downloaded to `data/images/`
3. Image encoded to base64
4. Sent to Ollama vision model with prompt
5. Vision model describes/analyzes image
6. Result returned to user

**Example**:
```python
from vision_handler import VisionHandler
from pathlib import Path

vision = VisionHandler(vision_model="llava")

# Analyze single image
result = vision.analyze_image(
    image_path=Path("screenshot.png"),
    user_prompt="What UI elements are visible?",
)

if result.success:
    print(result.description)

# Context-aware analysis
conversation_context = [
    {"role": "user", "content": "I'm building a login form"},
    {"role": "assistant", "content": "I can help with that!"},
]

result = vision.analyze_with_context(
    image_path=Path("form_mockup.png"),
    conversation_context=conversation_context,
)

# Compare two images
result = vision.compare_images(
    image1_path=Path("before.png"),
    image2_path=Path("after.png"),
    comparison_prompt="What changed between these screenshots?",
)
```

### 4. Bot Server Integration (`bot_server.py` - Modified)

**Changes Made**:

#### Added Imports (Lines 126-138):
```python
from tool_router import IntelligentToolRouter, RouteType, RouteDecision
from document_rag import DocumentRAGStore
from vision_handler import VisionHandler
INTELLIGENT_ROUTING_AVAILABLE = True  # (if imports succeed)
```

#### Global Variables (Lines 289-334):
```python
TOOL_ROUTER: Optional[Any] = None
RAG_STORE: Optional[Any] = None
VISION_HANDLER: Optional[Any] = None

def init_intelligent_components():
    """Initialize intelligent routing, RAG, and vision components"""
    # Initializes all three components with proper config
```

#### Document Handler (Lines 3779-3824):
```python
async def on_document(update, context):
    """Handle document uploads for RAG ingestion"""
    # Downloads file
    # Ingests into RAG store
    # Notifies user of success/failure
```

#### Photo Handler (Lines 3826-3873):
```python
async def on_photo(update, context):
    """Handle photo uploads for vision analysis"""
    # Downloads photo
    # Analyzes with vision model
    # Returns description
```

#### Intelligent Routing Helper (Lines 3879-3944):
```python
async def handle_intelligent_routing(text, user, chat, m, context):
    """Use intelligent routing to decide if special handling is needed"""
    # Analyzes intent
    # Routes to RAG, tools, or normal flow
    # Returns response if handled
```

#### Modified on_text Handler (Lines 3949-3979):
```python
async def on_text(update, context):
    # ... existing code ...

    # NEW: Try intelligent routing first (for admin users)
    if INTELLIGENT_ROUTING_AVAILABLE and user and user.id in ADMIN_WHITELIST:
        intelligent_response = await handle_intelligent_routing(text, user, chat, m, context)
        if intelligent_response:
            await m.reply_text(intelligent_response)
            return

    # ... existing conversation flow ...
```

#### Handler Registration (Lines 4355-4360):
```python
# Document and photo handlers (RAG + Vision)
app.add_handler(MessageHandler(filters.Document.ALL, on_document))
app.add_handler(MessageHandler(filters.PHOTO, on_photo))

# Text logger + reply logic
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Intelligent Agent System                         │
└─────────────────────────────────────────────────────────────────────┘

User Message
      ↓
┌─────────────────────┐
│  Message Handler    │
│  (bot_server.py)    │
└─────────────────────┘
      ↓
   Is Admin?
      ↓ Yes
┌──────────────────────────────────┐
│  IntelligentToolRouter           │
│  • Analyze intent                │
│  • Calculate heuristic scores    │
│  • Get LLM route scores          │
│  • Aggregate & select best       │
│  • Determine confidence          │
└──────────────────────────────────┘
      ↓
   Route Type?
      ↓
  ┌───────────────────────────────────────────┐
  │                                           │
  ↓ KNOWLEDGE_QUERY     ↓ WEB_SEARCH        ↓ IMAGE_ANALYSIS
┌──────────────────┐  ┌─────────────┐  ┌────────────────┐
│  DocumentRAGStore│  │  Existing   │  │  VisionHandler │
│  • Vector search │  │  Tool       │  │  • Encode      │
│  • Keyword FTS5  │  │  System     │  │  • Ollama API  │
│  • RRF merge     │  │             │  │  • Describe    │
└──────────────────┘  └─────────────┘  └────────────────┘
      ↓                     ↓                   ↓
   Response              Response            Response
      ↓                     ↓                   ↓
      └─────────────────────┴───────────────────┘
                          ↓
                    User receives
                      response
```

## How Components Work Together

### Example 1: Document Upload + Query

```
User: [uploads PDF "research.pdf"]
  ↓
bot_server.py: on_document handler
  ↓
DocumentRAGStore.add_document_from_telegram()
  ├─ Parse PDF with PyMuPDF
  ├─ Extract text from each page
  ├─ Split into ~1000 char chunks
  ├─ Generate embeddings (nomic-embed-text)
  ├─ Optional: Render pages to PNG
  ├─ Optional: Vision extract (llava)
  └─ Store in SQLite with vectors
  ↓
Bot: "✅ Document ingested successfully! Document ID: 3f4a8b21"

---

User: "What does the research say about climate change?"
  ↓
on_text → handle_intelligent_routing
  ↓
IntelligentToolRouter.analyze_intent()
  ├─ Heuristic: "document" keyword → KNOWLEDGE_QUERY +1.8
  ├─ LLM: Analyzes intent → KNOWLEDGE_QUERY 0.9
  ├─ Aggregate: KNOWLEDGE_QUERY wins with 3.6
  └─ Confidence: 0.82
  ↓
DocumentRAGStore.search("What does the research say about climate change?")
  ├─ Embed query → vector
  ├─ Vector similarity search → top 10 chunks
  ├─ FTS5 keyword search → top 10 chunks
  ├─ Reciprocal Rank Fusion → merge results
  └─ Return top 5 chunks with metadata
  ↓
Bot: "📚 Found 5 relevant passages:

1. Climate change research indicates... (page 12)
2. The study shows that... (page 34)
3. According to the findings... (page 56)

💡 From 1 document(s)"
```

### Example 2: Image Analysis

```
User: [uploads photo of UI mockup] "What do you think of this design?"
  ↓
bot_server.py: on_photo handler
  ↓
VisionHandler.analyze_image()
  ├─ Read image bytes
  ├─ Encode to base64
  ├─ Build Ollama messages:
  │   [
  │     {"role": "system", "content": "You are a helpful visual assistant..."},
  │     {"role": "user", "content": "What do you think of this design?", "images": ["base64..."]}
  │   ]
  ├─ Call ollama.chat(model="llava", ...)
  └─ Extract description from response
  ↓
Bot: "🖼️ Image Analysis:

This appears to be a login form UI mockup with a clean, modern design.
The form includes email and password fields, a 'Remember me' checkbox,
and a primary action button. The color scheme uses blue accents on a
white background, creating good contrast. The layout is centered and
well-spaced, following modern UI/UX best practices.

💬 Feel free to ask questions about this image!"
```

### Example 3: Web Search (Existing System)

```
User: "What are the latest Python 3.12 features?"
  ↓
on_text → handle_intelligent_routing
  ↓
IntelligentToolRouter.analyze_intent()
  ├─ Heuristic: "latest" keyword → WEB_SEARCH +1.5
  ├─ LLM: Analyzes intent → WEB_SEARCH 0.95
  ├─ Aggregate: WEB_SEARCH wins with 3.4
  └─ Confidence: 0.78
  ↓
Returns None (let existing system handle)
  ↓
Existing generate_agentic_reply flow
  ↓
Tools.search_internet("Python 3.12 features")
  ↓
Bot returns search results
```

## Configuration

### Environment Variables

Add to `.env`:

```bash
# Vision model for image analysis and PDF page extraction
OLLAMA_VISION_MODEL=llava

# Existing variables (required)
OLLAMA_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_URL=http://localhost:11434

# Admin users who get intelligent routing
ADMIN_WHITELIST=123456789,987654321
```

### Directory Structure

```
tg-auth-bot/
├── bot_server.py           # Main bot (modified)
├── tool_router.py          # NEW: Intelligent routing
├── document_rag.py         # NEW: RAG system
├── vision_handler.py       # NEW: Vision analysis
├── tool_integration.py     # Existing tool system
├── tool_schema.py          # Tool schemas
├── ai_tool_bridge.py       # Tool bridge
├── data/
│   ├── documents.db        # RAG database (auto-created)
│   ├── uploads/            # Uploaded documents (auto-created)
│   ├── images/             # Uploaded photos (auto-created)
│   └── .page_images/       # PDF page renders (auto-created)
└── examples/               # Reference implementation
    ├── main.py
    ├── context.py
    ├── tools.py
    └── ...
```

## Usage Examples

### As a User

#### Upload and Query Documents

```
User: [uploads research_paper.pdf]
Bot: "📄 Processing document: research_paper.pdf..."
Bot: "✅ Document ingested successfully!
Document ID: 3f4a8b21
You can now ask questions about this document."

User: "Summarize the key findings"
Bot: "📚 Found 5 relevant passages:

1. The study concludes that... (page 3)
2. Key findings include... (page 15)
3. Results demonstrate... (page 28)

💡 From 1 document(s)"
```

#### Analyze Images

```
User: [sends screenshot] "What's in this image?"
Bot: "🔍 Analyzing image..."
Bot: "🖼️ Image Analysis:

This screenshot shows a code editor with Python code visible.
The code appears to be implementing a neural network using PyTorch.
There's syntax highlighting enabled, and the file is named 'model.py'.
The code includes class definitions and method implementations.

💬 Feel free to ask questions about this image!"

User: "Can you explain the forward method?"
Bot: [Continues conversation about the image]
```

#### Web Search

```
User: "What's the weather in Tokyo?"
Bot: [Uses existing search tool]
Bot: "According to recent data, Tokyo currently has..."
```

### As a Developer

#### Check System Status

```python
# In Python console or script
from tool_router import IntelligentToolRouter
from document_rag import DocumentRAGStore
from vision_handler import VisionHandler

# Check if components are available
print(f"Router available: {IntelligentToolRouter is not None}")
print(f"RAG available: {DocumentRAGStore is not None}")
print(f"Vision available: {VisionHandler is not None}")
```

#### Test Routing

```python
from tool_router import IntelligentToolRouter

router = IntelligentToolRouter(ollama_model="llama3.2")

# Test intent analysis
decision = router.analyze_intent("Search for Python tutorials")
print(f"Route: {decision.route}")
print(f"Confidence: {decision.confidence}")
print(f"Reason: {decision.reason}")
print(f"Tools needed: {decision.tools_needed}")
```

#### Test RAG

```python
from document_rag import DocumentRAGStore
from pathlib import Path

rag = DocumentRAGStore(
    db_path=Path("data/documents.db"),
    embed_model="nomic-embed-text"
)

# Add a document
doc_id = rag.add_document_from_telegram(
    file_path=Path("test.pdf"),
    user_id=123,
    chat_id=456,
    message_id=789,
)

# Search
result = rag.search("machine learning", top_k=3)
for chunk in result.chunks:
    print(f"{chunk.text[:100]}...")
```

## Testing Checklist

### Prerequisites

- [ ] Ollama installed and running
- [ ] Models downloaded:
  - [ ] llama3.2 (or your chat model)
  - [ ] nomic-embed-text (or your embed model)
  - [ ] llava (or your vision model)
- [ ] Python dependencies installed:
  - [ ] PyMuPDF (fitz)
  - [ ] python-docx
  - [ ] chardet
  - [ ] beautifulsoup4
  - [ ] ollama

### Installation

```bash
# Install Python dependencies
pip install pymupdf python-docx chardet beautifulsoup4 ollama lxml

# Download Ollama models
ollama pull llama3.2
ollama pull nomic-embed-text
ollama pull llava
```

### Functional Tests

#### Test 1: Document Upload
- [ ] Upload a PDF file
- [ ] Bot confirms ingestion
- [ ] Ask a question about the document
- [ ] Receive relevant passages

#### Test 2: Image Analysis
- [ ] Upload a photo
- [ ] Bot analyzes and describes
- [ ] Ask follow-up question
- [ ] Bot responds with context

#### Test 3: Intelligent Routing
- [ ] Send message with "search for..."
- [ ] Verify WEB_SEARCH route selected
- [ ] Send message with "in the document..."
- [ ] Verify KNOWLEDGE_QUERY route selected

#### Test 4: Vision PDF Extraction
- [ ] Upload PDF with tables/figures
- [ ] Enable vision extraction (`use_vision=True`)
- [ ] Query about table data
- [ ] Verify vision-extracted content returned

### Performance Tests

- [ ] Upload large PDF (100+ pages)
- [ ] Measure ingestion time
- [ ] Query after ingestion
- [ ] Measure query response time

### Error Handling

- [ ] Upload unsupported file type
- [ ] Verify graceful error message
- [ ] Upload corrupted PDF
- [ ] Verify error handling
- [ ] Send image with Ollama offline
- [ ] Verify timeout handling

## Troubleshooting

### Issue: "Tool integration not available"

**Cause**: Dependencies not installed or import failed.

**Solution**:
```bash
pip install pymupdf python-docx chardet beautifulsoup4 ollama lxml
python3 -c "from tool_router import IntelligentToolRouter; print('OK')"
```

### Issue: "Vision analysis failed"

**Cause**: Vision model not available or Ollama not running.

**Solution**:
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Download vision model
ollama pull llava

# Test vision model
ollama run llava "describe this image" --image test.png
```

### Issue: "No relevant documents found"

**Cause**: No documents uploaded or query mismatch.

**Solution**:
1. Upload documents first
2. Wait for ingestion to complete
3. Try broader search terms
4. Check database:
```bash
sqlite3 data/documents.db "SELECT filename, status FROM documents;"
```

### Issue: "Routing confidence too low"

**Cause**: Ambiguous intent or low confidence threshold.

**Solution**:
Adjust threshold in router initialization:
```python
router = IntelligentToolRouter(
    min_confidence_threshold=0.2,  # Lower threshold
)
```

## Performance Optimization

### RAG Performance

**Slow ingestion**:
- Disable vision extraction for faster processing
- Use smaller chunk sizes
- Pre-process documents offline

**Slow queries**:
- Reduce `top_k` parameter
- Disable FTS5 if not needed
- Add indexes to database

### Vision Performance

**Slow image analysis**:
- Use smaller vision models (llava:7b vs llava:13b)
- Reduce image resolution before upload
- Use non-streaming mode

### Routing Performance

**Slow route selection**:
- Disable LLM routing, use heuristics only:
  ```python
  router = IntelligentToolRouter(enable_llm_routing=False)
  ```
- Cache routing decisions for similar queries
- Use faster models (qwen2.5:0.5b vs llama3.2:3b)

## Future Enhancements

### Phase 2: Enhanced Capabilities

- [ ] Multi-document queries (cross-document search)
- [ ] Document summarization
- [ ] Citation tracking (which doc, which page)
- [ ] Image-text combined search
- [ ] Conversation memory for image discussions

### Phase 3: Advanced Features

- [ ] Automatic document categorization
- [ ] Semantic caching for repeated queries
- [ ] Tool chaining (route → tool1 → tool2 → response)
- [ ] User feedback learning
- [ ] Per-user RAG databases

### Phase 4: Production Features

- [ ] Rate limiting
- [ ] Usage analytics
- [ ] Cost tracking (token usage)
- [ ] Multi-tenant isolation
- [ ] Backup and restore

## Code Statistics

### New Code

| File | Lines | Purpose |
|------|-------|---------|
| tool_router.py | 462 | Intelligent routing |
| document_rag.py | 701 | RAG system |
| vision_handler.py | 306 | Vision analysis |
| **Total** | **1,469** | **New code** |

### Modified Code

| File | Changes | Purpose |
|------|---------|---------|
| bot_server.py | +150 lines | Integration |

### Documentation

| File | Lines | Purpose |
|------|-------|---------|
| INTELLIGENT_AGENT_SYSTEM.md | This file | Complete guide |

**Total Implementation**: ~1,620 lines of code + documentation

## Integration with Existing Systems

### Tool Integration

The new routing system integrates with existing tool infrastructure:

```python
# Existing: tool_integration.py, tool_schema.py
# New: tool_router.py (uses existing tools)

# Flow:
IntelligentToolRouter.analyze_intent()
  ↓
decision.tools_needed = ["search_internet"]
  ↓
ToolExecutor.execute_tool("search_internet", ...)  # Existing
  ↓
Returns result
```

### Memory System

RAG complements existing memory:

```python
# Existing: Internal memory in bot_server.py
# New: DocumentRAGStore (separate knowledge base)

# User asks: "What did I say about Python?"
existing_memory → short-term conversation recall
RAG_STORE → long-term document knowledge

# Both systems work together
```

### Vision System

Vision handler extends existing multimodal capabilities:

```python
# Existing: Text-based conversation
# New: Image understanding

# User sends image + text
on_photo → Vision analysis
on_text → Conversation continues with image context
```

## Success Criteria

### ✅ Phase 1 Complete

- [x] Intelligent routing implemented
- [x] RAG system implemented
- [x] Vision handler implemented
- [x] All components integrated
- [x] Document/photo handlers added
- [x] Syntax validated (compiles without errors)
- [x] Architecture documented

### ⏳ Phase 2 (Testing Required)

- [ ] Document upload → ingestion → query works end-to-end
- [ ] Image upload → analysis → response works
- [ ] Routing selects correct route for different intents
- [ ] Performance acceptable (< 5s for queries)
- [ ] Error handling works gracefully

### 📋 Phase 3 (Production Ready)

- [ ] All tests passing
- [ ] Performance optimized
- [ ] Error rates < 1%
- [ ] User feedback positive
- [ ] Documentation complete

## Conclusion

✅ **The intelligent agent system is fully implemented and ready for testing.**

The system successfully integrates:
- Automatic tool selection based on user intent
- Document RAG with hybrid retrieval
- Vision analysis with Ollama multimodal models
- Seamless integration with existing bot infrastructure

**Next Steps**:
1. Start the bot: `python bot_server.py`
2. Upload a document and ask questions
3. Send images for analysis
4. Monitor logs for routing decisions
5. Adjust configuration as needed

All components are modeled after the `/examples/` reference implementation and follow best practices for agentic LLM systems.

---

**Implementation Date**: 2025-10-19
**Total Lines**: ~1,620 (code + docs)
**Status**: ✅ Ready for Testing
**Session**: Intelligent Agent Integration Complete
