# MCP Integration for QLORAX Enhanced

This directory contains the Model Context Protocol (MCP) server integration for the QLORAX Enhanced project. The MCP integration provides enhanced AI capabilities with GitHub integration without affecting the core codebase.

## 🌟 Features

- **GitHub Integration**: Direct repository management, issue tracking, and PR workflows
- **VS Code Integration**: Seamless development experience with AI assistance
- **Non-intrusive**: Completely separate from core QLORAX functionality
- **Production Ready**: Configurable for development and production environments

## 🚀 Quick Setup

### 1. Install Required VS Code Extensions

The following extensions provide MCP functionality:

```bash
# Core MCP extensions
code --install-extension automatalabs.copilot-mcp
code --install-extension semanticworkbenchteam.mcp-server-vscode
code --install-extension zebradev.mcp-server-runner

# GitHub integration
code --install-extension ms-azuretools.vscode-azure-mcp-server
```

### 2. Configure MCP Servers

```bash
# Copy configuration files
cp mcp/config/mcp-settings.json ~/.vscode/settings.json

# Set up GitHub token (optional - for enhanced GitHub features)
export GITHUB_TOKEN="your_github_token_here"
```

### 3. Start MCP Servers

```bash
# From the mcp directory
cd mcp
python servers/github_mcp_server.py

# Or use the VS Code MCP Server Runner extension
```

## 📁 Directory Structure

```
mcp/
├── README.md                 # This file
├── config/
│   ├── mcp-settings.json    # VS Code MCP configuration
│   ├── server-config.yaml   # MCP server configuration
│   └── github-config.json   # GitHub integration settings
├── servers/
│   ├── github_mcp_server.py # Custom GitHub MCP server
│   ├── qlorax_mcp_server.py # QLORAX-specific MCP tools
│   └── utils.py             # Utility functions
└── docs/
    ├── setup-guide.md       # Detailed setup instructions
    ├── usage-examples.md    # Usage examples and workflows
    └── troubleshooting.md   # Common issues and solutions
```

## 🔧 Configuration

### GitHub Integration

The GitHub MCP server provides:
- Repository management
- Issue and PR tracking
- Code review assistance
- Automated workflows
- Release management

### QLORAX Integration

The QLORAX MCP server provides:
- Training job monitoring
- Model performance tracking
- Experiment management
- Result analysis
- Documentation generation

## 🎯 Usage Examples

### Basic GitHub Operations

```python
# Create a new issue
mcp.github.create_issue(
    title="Enhance model accuracy",
    body="Investigation needed for improving BERT F1 scores",
    labels=["enhancement", "model-performance"]
)

# Create a pull request
mcp.github.create_pr(
    title="Add new evaluation metrics",
    branch="feature/new-metrics",
    base="main"
)
```

### QLORAX Operations

```python
# Monitor training progress
status = mcp.qlorax.get_training_status()

# Generate performance report
report = mcp.qlorax.generate_performance_report(
    model_path="models/latest",
    metrics=["bert_score", "rouge", "bleu"]
)
```

## 🔗 Integration with Main Project

The MCP integration is designed to work alongside your existing QLORAX Enhanced project:

- **Non-intrusive**: No changes to core codebase
- **Complementary**: Enhances development workflow
- **Optional**: Can be disabled without affecting main functionality
- **Extensible**: Easy to add new MCP tools and integrations

## 📚 Documentation

- [Setup Guide](docs/setup-guide.md) - Detailed installation and configuration
- [Usage Examples](docs/usage-examples.md) - Common workflows and examples  
- [Troubleshooting](docs/troubleshooting.md) - Solutions for common issues

## 🤝 Contributing

MCP integration contributions are welcome! Please follow the same contributing guidelines as the main project.

## 📄 License

Same MIT License as the main QLORAX Enhanced project.