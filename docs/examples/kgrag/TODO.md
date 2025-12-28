# KGRag Example - TODO List

This document outlines improvements needed for the KGRag example to make it more accessible and useful for Mellea users.

## 1. Documentation Improvements

### 1.1 Missing README.md
**Priority: High** ✅ **COMPLETED**

Create a comprehensive README.md that includes:
- [x] What KGRag is and what problem it solves
- [x] Architecture overview and how it uses Mellea
- [x] Prerequisites (Neo4j, required packages, dataset)
- [x] Step-by-step setup instructions
- [x] How to run the example (both preprocessing and QA)
- [x] Expected outputs and how to interpret results
- [x] Troubleshooting common issues

### 1.2 Incomplete Setup Instructions
**Priority: High** ✅ **COMPLETED**

- [x] Add clear guide on setting up Neo4j database
  - [x] Installation instructions
  - [x] Required plugins (APOC)
  - [x] Vector index creation
- [x] Provide instructions for obtaining/preparing the movie dataset
  - [x] Either include sample dataset or provide download link
  - [x] Document expected data format
- [x] Create comprehensive .env_template with all required variables
- [x] Add example configurations for different backends (OpenAI, Ollama, local models)

### 1.3 Configuration Issues
**Priority: High** ✅ **COMPLETED**

Fix hardcoded configurations in `constants.py`:
- [x] Remove IBM-specific endpoints that external users cannot access
- [x] Remove local model paths (`/net/storage149/...`)
- [x] Fix duplicate variable assignments
- [x] Move all configuration to environment variables
- [x] Provide sensible defaults that work out-of-the-box
- [ ] Add configuration validator to check required variables

**Recent Changes:**
- Converted all files to use environment variables with `os.getenv()` and `python-dotenv`
- Removed direct imports from `constants.py` in all 8 files
- Created comprehensive `.env_template` with all configuration options
- Added default values for all environment variables

### 1.4 Dependencies Documentation
**Priority: High** ✅ **COMPLETED**

- [x] Create requirements.txt or update pyproject.toml with all dependencies:
  - [x] neo4j driver
  - [x] sentence-transformers (optional)
  - [x] python-dotenv
  - [x] tqdm
  - [x] Other required packages
- [x] Document Neo4j version requirements
- [x] Document APOC plugin requirement
- [x] Add installation instructions for each dependency

## 2. Code Quality Issues

### 2.1 kg_model.py Improvements
**Priority: Medium**

- [ ] Fix duplicate `import asyncio` at lines 1-2
- [ ] Refactor 1150+ line file into smaller modules:
  - [ ] `prompts.py` - All prompt templates
  - [ ] `kg_model.py` - Core KGModel class
  - [ ] `query_processing.py` - Query breakdown and entity extraction
  - [ ] `graph_search.py` - Graph traversal logic
- [ ] Move prompts to separate Jinja2 templates in `templates/` directory
- [ ] Remove commented-out code at line 174
- [ ] Document magic numbers (`route=5, width=30, depth=3`) in docstrings or config
- [ ] Add type hints to all functions
- [ ] Add docstrings to all public methods

### 2.2 run_qa.py Improvements
**Priority: Medium**

- [ ] Fix global `participant_model` usage - pass as parameter
- [ ] Fix undefined `args.dataset` in logging (line 212)
- [ ] Fix undefined `args.model` in logging (line 212)
- [ ] Remove duplicate `torch` import (lines 24 and 133)
- [ ] Add better error handling and validation
- [ ] Add progress logging improvements
- [ ] Separate concerns: data loading, model inference, evaluation

### 2.3 run.sh Improvements
**Priority: Low**

- [ ] Complete the incomplete comment reference (lines 2-4)
- [ ] Add error handling (set -e)
- [ ] Add usage instructions in comments
- [ ] Make parameters configurable via command-line arguments
- [ ] Add example usage with different configurations

### 2.4 eval.py Improvements
**Priority: Medium** ✅ **PARTIALLY COMPLETED**

- [x] Document evaluation metrics and scoring system
- [ ] Move hardcoded prompts and examples to configuration
- [x] Add explanation of LLM-as-judge approach
- [x] Provide guidance on interpreting evaluation results
- [ ] Add visualization of results

## 3. Missing Integration with Mellea Patterns

### 3.1 Demonstrate Mellea Features
**Priority: High** ✅ **PARTIALLY COMPLETED**

- [ ] Add examples using `@generative` decorator for entity extraction
- [ ] Use `RejectionSamplingStrategy` for validation loops
- [ ] Showcase Mellea's requirement validation features
- [ ] Demonstrate `m.instruct()` and `m.chat()` methods directly
- [x] Show how to use ModelOption for backend-agnostic configuration
- [ ] Add examples of context management
- [ ] Demonstrate sampling strategies for improving output quality

Note: README now documents Mellea integration patterns and provides examples of MelleaSession and backend configuration.

### 3.2 Align with Mellea Philosophy
**Priority: Medium**

- [ ] Refactor to show "generative programming" concepts clearly
- [ ] Add instruct-validate-repair pattern examples
- [ ] Show how KG-RAG fits into generative programming paradigm
- [ ] Demonstrate composability with other Mellea components

## 4. Testing and Validation

### 4.1 Add Test Files
**Priority: Medium**

- [ ] Unit tests for KG operations (entity/relation CRUD)
- [ ] Unit tests for query processing functions
- [ ] Integration tests for end-to-end QA pipeline
- [ ] Example test cases with expected outputs
- [ ] Mock Neo4j for faster testing

### 4.2 Add Example Outputs
**Priority: Low**

- [ ] Include sample query results
- [ ] Show example graph visualizations
- [ ] Provide benchmark results

## 5. Structural Improvements

### 5.1 Simplify Entry Point
**Priority: High**

- [ ] Create a simpler starting point (e.g., `simple_example.py`)
- [ ] Add step-by-step tutorial progression:
  1. Basic KG creation
  2. Simple entity/relation queries
  3. Basic QA with KG
  4. Full KG-RAG pipeline
- [ ] Explain each component's role before integration
- [ ] Compare to simpler approaches (e.g., vanilla RAG)

### 5.2 Better Module Organization
**Priority: Medium**

Suggested structure:
```
kgrag/
├── README.md
├── TODO.md
├── requirements.txt
├── .env_template
├── data/
│   ├── README.md (how to obtain dataset)
│   └── sample_data.jsonl (small sample)
├── configs/
│   ├── default_config.yaml
│   ├── openai_config.yaml
│   └── ollama_config.yaml
├── src/
│   ├── __init__.py
│   ├── kg_model.py
│   ├── prompts.py (or templates/)
│   ├── query_processing.py
│   ├── graph_search.py
│   └── evaluation.py
├── kg/
│   ├── __init__.py
│   ├── kg_driver.py
│   ├── kg_rep.py
│   └── kg_preprocessor.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── utils.py
│   └── data.py
├── examples/
│   ├── simple_kg_example.py
│   ├── basic_qa_example.py
│   └── full_pipeline_example.py
├── tests/
│   ├── test_kg_operations.py
│   ├── test_query_processing.py
│   └── test_evaluation.py
├── notebooks/
│   └── kgrag_tutorial.ipynb
├── scripts/
│   ├── setup_neo4j.sh
│   ├── preprocess_data.sh
│   └── run_evaluation.sh
└── run.sh (main entry point)
```

## 6. Educational Content

### 6.1 Add Tutorial Content
**Priority: Medium**

- [ ] Create Jupyter notebook with step-by-step tutorial
- [ ] Add inline documentation explaining the KG-RAG approach
- [ ] Include diagrams showing data flow
- [ ] Add links to relevant papers/resources
- [ ] Explain when to use KG-RAG vs. other approaches

### 6.2 Add Comparison Examples
**Priority: Low**

- [ ] Compare performance with vanilla RAG
- [ ] Show accuracy improvements with KG enhancement
- [ ] Demonstrate token efficiency benefits
- [ ] Provide cost analysis

## 7. Deployment and Scalability

### 7.1 Production Readiness
**Priority: Low**

- [ ] Add Docker configuration
- [ ] Add docker-compose for full stack (Neo4j + app)
- [ ] Add monitoring and logging best practices
- [ ] Add error recovery mechanisms
- [ ] Add rate limiting and retry logic

### 7.2 Performance Optimization
**Priority: Low**

- [ ] Profile and optimize bottlenecks
- [ ] Add caching strategies
- [ ] Document scaling considerations
- [ ] Add batch processing examples

## 8. Accessibility Improvements

### 8.1 Make It Beginner-Friendly
**Priority: High** ✅ **PARTIALLY COMPLETED**

- [x] Add "Quick Start" section with minimal setup
- [ ] Provide pre-populated sample database option
- [ ] Add FAQ section
- [ ] Include video walkthrough (optional)
- [x] Add glossary of terms (KG, RAG, triplets, etc.)

Note: README includes comprehensive beginner-friendly content with clear explanations and examples.

### 8.2 Cross-Platform Support
**Priority: Medium**

- [ ] Test on Windows, macOS, Linux
- [ ] Document platform-specific issues
- [ ] Provide platform-specific setup instructions
- [ ] Add GitHub Actions CI for testing

## Priority Summary

**Immediate (High Priority):**
1. ✅ Create README.md with setup instructions - **COMPLETED**
2. ✅ Fix configuration issues (remove hardcoded IBM endpoints) - **COMPLETED**
3. ✅ Document dependencies and prerequisites - **COMPLETED**
4. ⚠️ Show Mellea pattern integration - **PARTIALLY COMPLETED**

**Short-term (Medium Priority):**
5. Refactor code for better organization
6. Add tests and examples
7. Improve code quality (remove duplicates, add docs)
8. Add tutorial content

**Long-term (Low Priority):**
9. Add deployment configurations
10. Performance optimization
11. Comparison studies
12. Platform testing

## Recent Progress (2024)

### Completed Items:
- ✅ **Comprehensive README.md created** with:
  - Overview and problem statement
  - Architecture diagram and component descriptions
  - Detailed setup instructions for Neo4j
  - Environment configuration examples
  - Step-by-step usage guide
  - Example queries and code snippets
  - Troubleshooting section
  - Performance optimization tips
  - Integration with Mellea framework documentation

### Next Priority Items:
1. ✅ **Configuration cleanup**: Remove IBM-specific endpoints and hardcoded paths in `constants.py` - **COMPLETED**
2. ✅ **Create .env_template**: Template file with all required environment variables - **COMPLETED**
3. **Enhanced Mellea integration**: Add more examples using `@generative`, requirements, and sampling strategies
4. **Code refactoring**: Break down large files (especially `kg_model.py`) into smaller modules
5. **Add configuration validator**: Check required variables on startup

## Notes

- Focus on making this example accessible to external users first
- Prioritize documentation over new features ✅ **Making good progress!**
- Ensure consistency with other Mellea examples
- Get feedback from users early and iterate
- README now provides comprehensive documentation for external users
