# 🧠 NeuroShell

**The intelligent terminal-based browser that try to eliminates the need for conventional web browsing.**

NeuroShell is a command-line AI browser that combines local language models with real-time web search to deliver comprehensive, contextual answers directly in your terminal. Say goodbye to opening multiple browser tabs and hello to instant, intelligent responses.

## 🎯 Project Vision

NeuroShell was born from a simple yet powerful idea: **Why open a browser when you can get better answers faster?**

Instead of:
- Opening a browser
- Typing in search engines
- Clicking through multiple links
- Reading through irrelevant content
- Switching between tabs

You get:
- Instant intelligent responses
- Contextual AI analysis
- Curated information from multiple sources
- Beautiful terminal interface
- Conversation memory

## ✨ Features

- **🧠 Neural Intelligence**: Seamlessly integrates with Ollama for local language model inference
- **🔍 Multi-Provider Web Search**: Supports Google and Brave search engines, and Tavily for web page extraction
- **🎯 Query Optimization**: Automatically optimizes user queries for better search results
- **📄 Intelligent Content Extraction**: Extracts and processes relevant content from web pages
- **💬 Conversational Memory**: Maintains conversation history for contextual responses
- **🎨 Rich Terminal Interface**: Beautiful, interactive terminal UI with typewriter effects
- **⚡ Async Architecture**: High-performance asynchronous processing
- **🛡️ Robust Error Handling**: Comprehensive error handling and graceful degradation
- **🔧 Configurable**: Flexible configuration options for timeouts, content limits, and more
- **🚀 Browser Replacement**: Complete terminal-based browsing experience

## 🏗️ Architecture Overview

NeuroShell follows a modular, neural-inspired architecture that processes queries like a human brain:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │ -> │ Neural Processor │ -> │  Web Search     │
│   (Terminal)    │    │  (Optimization)  │    │ (Multi-Source)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                |
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Smart Response  │ <- │ Content Synthesis│ <- │ Content Extract │
│   (Terminal)    │    │   (AI Analysis)  │    │  (Web Pages)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Neural Components

- **AppState**: Central nervous system for application components
- **ConversationManager**: Orchestrates the complete neural processing pipeline
- **ModelManager**: Handles Ollama model selection and neural network management
- **SearchResultProcessor**: Processes and synthesizes search results
- **InputValidator**: Validates and sanitizes user input
- **Configuration**: Centralized neural configuration management

## 📋 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Ollama**: Installed and running locally
- **Internet Connection**: Required for web search functionality
- **Terminal**: Modern terminal with Unicode support (recommended)

## 🚀 Installation

### Step 1: Install Ollama (The Neural Engine)

Follow the [official Ollama installation guide](https://ollama.ai/download) for your operating system.

**Start the neural engine:**
```bash
# On macOS/Linux
ollama serve

# Or if using system service
systemctl start ollama
```

**Install a neural model:**
```bash
# Recommended: Install Gemma 3 model
ollama pull gemma3

# For faster responses: Install Phi3
ollama pull phi3

# For advanced users: Install larger models
ollama pull llama2:13b
```

### Step 2: Clone and Setup NeuroShell

```bash
# Clone NeuroShell
git clone https://github.com/thejasmeetsingh/neuro-shell neuroshell
cd neuroshell

# Create neural environment
python -m venv venv

# Activate neural environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install neural dependencies
pip install -r requirements.txt
```

### Step 3: Neural Configuration

Create a `.env` file in the project root with your search API keys:

```env
# Search API Keys (configure at least one for optimal performance)
BRAVE_API_KEY=your_brave_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_google_cse_id_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### Step 4: Verify Neural Installation

```bash
# Test neural engine connection
curl http://localhost:11434/api/tags

# Start NeuroShell
python main.py
```

## 🎮 Usage

### Basic Neural Browsing

1. **Launch NeuroShell:**
   ```bash
   python main.py
   ```

2. **Select your neural model** from the available options

3. **Start browsing with your brain!** Ask anything and get intelligent responses

### Example Neural Session


### Neural Commands

- **Regular queries**: Just ask naturally like you're talking to a smart assistant
- **Exit commands**: `exit`, `quit`, `q`, or `Ctrl+C`
- **Context**: NeuroShell remembers your conversation for better responses


### Neural Search Provider Setup

#### Brave Search (Recommended)
1. Sign up at [Brave Search API](https://api.search.brave.com/)
2. Get your API key from the dashboard
3. Add to `.env` file

#### Google Custom Search
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the Custom Search API
3. Create credentials and get your API key
4. Set up a Custom Search Engine at [Google CSE](https://cse.google.com/)

#### Tavily Search (New!)
1. Sign up at [Tavily AI](https://tavily.com/)
2. Get your API key from the dashboard
3. Add to `.env` file

## 🧠 Advanced Neural Features

### Neural Query Processing Pipeline

1. **Neural Input Validation**: Validates query format and intent
2. **Neural Query Optimization**: Uses AI to enhance search effectiveness
3. **Multi-Source Neural Search**: Performs intelligent search across providers
4. **Neural Result Ranking**: AI determines most relevant information sources
5. **Neural Content Extraction**: Extracts and processes detailed content
6. **Neural Response Synthesis**: Creates comprehensive, contextual responses

### Extending NeuroShell

1. **New Search Providers**: Extend neural search capabilities
2. **Custom Neural Prompts**: Add specialized neural behaviors
3. **Neural Output Formats**: Create new response formats
4. **Neural Configuration**: Add advanced neural parameters

## 🔍 Troubleshooting

### Common Neural Issues

#### "No neural models available"
```bash
# Install a neural model
ollama pull llama2

# Verify neural installation
ollama list
```

#### "Neural engine not responding"
```bash
# Start neural engine
ollama serve

# Check neural status
curl http://localhost:11434/api/tags
```

#### "No neural search results"
- Verify API keys in `.env` file
- Check internet connection for neural web access
- Try different neural query formulations

### Neural Performance Optimization

- **Model Selection**: Smaller neural models (Phi3 or Gemma3) = faster responses
- **Content Limits**: Adjust neural content processing limits
- **Search Count**: Optimize neural search result count
- **Timeout Settings**: Tune neural timeouts for your hardware
