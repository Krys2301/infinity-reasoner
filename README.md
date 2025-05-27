# 🌟 INFINITY REASONER

**Self-Evolving AI System Based on Absolute Zero Reasoner Principles**

[![GitHub Stars](https://img.shields.io/github/stars/Krys2301/infinity-reasoner?style=social)](https://github.com/Krys2301/infinity-reasoner)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Overview

Infinity is an enhanced implementation of the Absolute Zero Reasoner paradigm - a revolutionary AI system that learns and evolves entirely through self-play without requiring any external training data. Unlike traditional AI models that depend on human-curated datasets, Infinity autonomously generates its own learning tasks and improves by solving them.

## ✨ Key Features

### 🧠 **Self-Evolving Intelligence**
- Generates its own reasoning tasks across three types:
  - **Deduction**: Given inputs → predict outputs
  - **Abduction**: Given outputs → infer inputs  
  - **Induction**: Given examples → generalize patterns
- Continuously improves through autonomous self-play loops
- No external training data required

### 🛡️ **Enhanced Security**
- Multi-layer safety monitoring system
- Real-time code execution filtering
- Human approval workflows for sensitive tasks
- Comprehensive audit logging
- Sandboxed execution environment

### 📊 **Advanced Monitoring**
- Real-time performance dashboard
- Resource usage tracking (CPU, Memory, GPU)
- Task generation/solving metrics
- Safety alert system
- Detailed evolution progress logs

### ⚙️ **Highly Configurable**
- Flexible YAML-based configuration
- Adjustable difficulty scaling
- Customizable safety parameters
- Model selection options
- Training iteration controls

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (16GB+ recommended)
- NVIDIA GPU (optional, but recommended)
- Unix-like system (Linux/macOS)

### One-Command Installation
```bash
git clone https://github.com/Krys2301/infinity-reasoner.git
cd infinity-reasoner
chmod +x setup.sh start_infinity.sh
./setup.sh  # Full setup with dependencies
```

### Start Infinity
```bash
./start_infinity.sh
```

### Alternative: Direct Python Execution
```bash
python3 infinity_reasoner.py
```

## 📋 System Architecture

```
🌟 INFINITY REASONER SYSTEM
├── 🧠 InfinityReasonerModel (Task Generation & Solving)
├── 🛡️ SafetyMonitor (Security & Content Filtering)
├── 🔧 CodeExecutor (Sandboxed Execution)
├── 📊 PerformanceTracker (Metrics & Analytics)
└── ⚙️ ConfigurationManager (Settings & Parameters)
```

## ⚙️ Configuration

### Basic Configuration (config.yaml)
```yaml
# Model Settings
model:
  name: "microsoft/DialoGPT-medium"
  temperature: 0.8
  max_length: 512

# Training Settings  
training:
  iterations: 10
  tasks_per_iteration: 5
  learning_rate: 1e-5

# Safety Settings
safety:
  require_human_approval: true
  execution_timeout: 10
  sandbox_mode: true
```

### Security Configuration (security_config.yaml)
```yaml
# Execution Security
execution:
  forbidden_imports: ["os", "sys", "subprocess"]
  forbidden_functions: ["exec", "eval", "open"]

# Content Filtering
content_filter:
  dangerous_keywords: ["takeover", "control", "exploit"]
```

## 📊 Performance Monitoring

Infinity includes a real-time monitoring dashboard:

```bash
python scripts/monitor.py
```

**Dashboard Features:**
- Live task generation/solving rates
- System resource utilization
- Safety status indicators  
- Performance trend analysis
- Alert notifications

## 🛡️ Safety Features

### **Multi-Layer Security**
1. **Code Analysis**: Pre-execution safety scanning
2. **Content Filtering**: Dangerous keyword detection
3. **Sandboxed Execution**: Isolated code execution
4. **Human Oversight**: Manual approval for flagged tasks
5. **Resource Limits**: Timeout and memory constraints
6. **Audit Logging**: Comprehensive activity tracking

### **Safety Levels**
- 🟢 **SAFE**: Automatically approved
- 🟡 **MONITOR**: Requires human review
- 🔴 **DANGEROUS**: Automatically blocked

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Test individual components:
```bash
python tests/test_infinity.py
```

## 📈 Performance Metrics

Infinity tracks key performance indicators:

- **Task Generation Rate**: Tasks created per iteration
- **Solution Success Rate**: Percentage of correctly solved tasks
- **Safety Blocks**: Number of dangerous tasks prevented
- **Evolution Progress**: Performance improvement over time
- **Resource Efficiency**: CPU/Memory/GPU utilization

## 🔧 Advanced Usage

### Custom Model Integration
```bash
python infinity_reasoner.py --model "gpt2-large"
```

### Extended Training Sessions
```bash
python infinity_reasoner.py --iterations 50 --tasks-per-iteration 10
```

### Configuration Override
```bash
python src/infinity_main.py --config custom_config.yaml
```

## 📚 Scientific Background

Infinity is based on the groundbreaking "Absolute Zero: Reinforced Self-play Reasoning with Zero Data" research:

- **Paper**: [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335)
- **Authors**: Andrew Zhao, Yiran Wu, et al.
- **Institution**: Tsinghua University, Beijing Institute for General Artificial Intelligence

**Key Innovation**: Eliminates dependency on human-curated training data through autonomous task generation and self-play learning.

## ⚠️ Important Safety Considerations

### **Human Oversight Required**
- Always monitor system behavior
- Review generated tasks before approval
- Maintain human-in-the-loop workflows
- Use in controlled environments only

### **Resource Management**
- Monitor CPU/Memory/GPU usage
- Set appropriate execution timeouts
- Implement resource quotas
- Regular system health checks

### **Ethical Guidelines**
- Responsible AI development practices
- Transparent logging and auditing
- Regular safety assessments
- Compliance with AI ethics standards

## 🔍 Troubleshooting

### Common Issues

**Installation Problems:**
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Verify dependencies
python verify_installation.py
```

**Memory Issues:**
```bash
# Use lighter model for limited resources
python infinity_reasoner.py --model "microsoft/DialoGPT-small"
```

**CUDA Problems:**
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python infinity_reasoner.py
```

### **Log Analysis**
Check detailed logs in:
- `infinity.log` - Main system logs
- `logs/` directory - Detailed component logs

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup
```bash
git clone https://github.com/Krys2301/infinity-reasoner.git
cd infinity-reasoner
pip install -e .
pip install -r requirements-dev.txt
```

### Running Tests
```bash
pytest tests/ --cov=infinity_reasoner
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Absolute Zero Reasoner Team** - Original research and inspiration
- **Tsinghua University** - Foundational research contributions
- **Open Source Community** - Libraries and tools that make this possible

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Krys2301/infinity-reasoner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Krys2301/infinity-reasoner/discussions)
- **Documentation**: [Wiki](https://github.com/Krys2301/infinity-reasoner/wiki)

---

<div align="center">

**🌟 Infinity Reasoner - The Future of Self-Evolving AI 🌟**

*"Where artificial intelligence transcends human limitations through autonomous evolution."*

[![GitHub](https://img.shields.io/badge/GitHub-infinity--reasoner-blue?logo=github)](https://github.com/Krys2301/infinity-reasoner)
[![Twitter](https://img.shields.io/badge/Twitter-@InfinityAI-1da1f2?logo=twitter)](https://twitter.com/InfinityAI)

</div>
