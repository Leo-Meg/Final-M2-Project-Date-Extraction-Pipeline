# Final-M2-Project-Date-Extraction-Pipeline

This repository contains the implementation of an automated date extraction system for French administrative documents, developed as part of the NLP in Industry final project at Université Paris Cité M2 NLP. I do this work with three classmates : Xu Sun, Dan Hou and Haeeul Hwang.



## Table of contents

1. [Pipeline Components](#pipeline)
   - [Step 1 – Data Preprocessing](#step1)
   - [Step 2 – Data Entity Extraction](#step2)
   - [Step 3 – LLM-based Data Selection](#step3)
   - [Step 4 – Date Format Cleaning](#step4)
   - [Evaluation](#eval)
2. [Environment Setup](#env)
   - [Option 1 : Automatic Setup](#aut)
   - [Option 2 : Manuel Setup](#man)
3. [System Requirements](#syst)
   - [Important Notes](#not)
   - [Troubleshooting](#troubl)
   - [Performance Benchmarks](#perf)
4. [Tools](#tools)
   - [Models](#mod)
   - [Data](#dat)
   - [Device](#dev)

Our pipeline aims to efficiently extract dates from French administrative documents through several key steps:

<a name="pipeline"></a>
## 1. Pipeline Components

<a name="step1"></a>
### Step 1. Data Preprocessing (`1_dataset_rebuild.py`)
The first step focuses on preparing high-quality data for NER and LLM processing:

- **Asynchronous Data Collection**
  - Downloads content from provided URLs using ]
  - Validates file accessibility and content length (>500 chars)
  - Handles parallel downloads efficiently

- **Text Normalization**
  - Concatenates source URL with document content
  - Removes redundant whitespace and special characters
  - Creates two versions of content:
      - `text_content`: URL + normalized text
      - `raw_text_content`: Only normalized text

- **Output**
- Generates `dataset_valid.csv` containing:
    - Original metadata
    - Local file paths
    - Normalized text content
    - Raw text content

<a name="step2"></a>
### Step 2. Date Entity Extraction (`2_ner.py`)
Extracts candidate dates using NER:

- **Model**: Utilizes CamemBERT-NER (fine-tuned for French date extraction)

- **Optimization Features**:
  - GPU-accelerated batch processing
  - Dynamic text chunking for long documents
  - Efficient memory management
  - Multi-worker data loading

- **Processing Steps**:
  - Tokenizes and processes text in batches
  - Identifies date entities
  - Filters and validates dates
  - Maintains extraction context

<a name="step3"></a>
### Step 3. LLM-based Date Selection (`4_llm_reference.py`)
Uses LLM reasoning to select the most accurate publication date:

- **Model**: Qwen2 5.7B/14B
  - Instruction tuned for natural language understanding
  - Enhanced context window (80k tokens) with the help of Vllm and rope-scaling

- **Selection Process**:
  - Takes NER extracted dates as candidates
  - Uses few-shot learning with carefully selected examples
  - Processes full document context
  - Returns single most likely publication date

- **Key Features**:
  - VLLM acceleration for fast inference
  - Efficient batching and memory usage
  - Robust error handling
  - Context-aware date selection

<a name="step4"></a>
### 4. Date Format Cleaning (clean_date.py)
Standardizes extracted dates into a uniform format:

- **Date Pattern Recognition**

  - Handles multiple French date formats:

    * DD Month YYYY (e.g., "1er juillet 2023")
    * DD/MM/YYYY or DD/MM/YY
    * DD-MM-YYYY or DD-MM-YY
    * Month YYYY (e.g., "OCTOBRE 2022")


  - **Supports variations in month names (lowercase, uppercase, accented)**
  
  
  - **Format Standardization**

    * Converts all dates to DD/MM/YYYY format
    * Handles French month names using comprehensive mapping
    * Processes special cases like "1er" (first of month)
    * Assumes 20xx for two-digit years
    * Sets default day to 01 for month-year only dates

<a name="eval"></a>
### 6. Evaluation (6_evaluation.py)
   Assesses the accuracy of date extraction results:

- **Accuracy Metrics**

  *   Compares extracted dates with gold standard labels
  *   Calculates two accuracy scores:
      1. Datapolitics accuracy (published vs. gold label)
      2. Our prediction accuracy (cleaned prediction vs. gold label)

  **Output Generation**
    *   Creates comprehensive evaluation report
    *   Includes metadata and comparison columns
    *   Saves results in CSV format


  - **Key Features**
    *   Handles missing columns gracefully
    *   Supports flexible input/output paths
    *   Provides formatted accuracy statistics

<a name="env"></a>  
## 2. Environment Setup


We provide two ways to set up the environment:
<a name="aut"></a>
#### Option 1: Automatic Setup (Recommended)
Run our setup script to automatically create and configure the environment:
```bash
# Make the script executable
chmod +x pipeline_environment.sh
```

# Run the setup script
```bash 
./pipeline_environment.sh
```
<a name="man"></a>
#### Option 2: Manuel Setup (NOT Recommended)
If you prefer to set up manually, follow these steps:

1. Create and activate conda environment:

```bash 
# Create new environment
conda create -n automated_date_extraction python=3.12
conda activate automated_date_extraction

# Install CUDA toolkit
conda install cuda-cudart=12.1.105=0 -c nvidia

# Install PyTorch with CUDA support
conda install pytorch=2.3.0=py3.12_cuda12.1_cudnn8.9.2_0 -c pytorch

# Install dependencies
pip install ninja
pip install flash-attn --no-build-isolation 
pip install modelscope==1.18.0
pip install openai==1.46.0
pip install tqdm==4.66.2
pip install transformers==4.44.2
pip install vllm==0.6.1.post2

# Download Qwen models
modelscope download Qwen/Qwen2.5-14B-Instruct
modelscope download Qwen/Qwen2.5-7B-Instruct
```

<a name="syst"></a>
### System Requirements
* **Operating System**: Ubuntu 22.04 or compatible Linux distribution
* **GPU**: NVIDIA GPU with
  * Minimum 24GB VRAM for Qwen 7B model
  * Minimum 40GB VRAM for Qwen 14B model
* **CUDA**: CUDA 12.1
* **Package Manager**: Conda
  
<a name="not"></a>
### Important Notes
* **Windows Compatibility**: Not guaranteed due to flash-attention dependencies
* **Flash Attention**: For specific installation requirements of flash-attention (v2.6.3), please refer to the [official documentation](https://github.com/Dao-AILab/flash-attention)
* **GPU Memory**: Ensure sufficient GPU memory is available before running the models
* **Installation Issues**: If you encounter issues with flash-attention installation, try installing it separately after other dependencies

<a name="troubl"></a>
### Troubleshooting
If you encounter any installation issues:
1. Ensure your CUDA drivers are properly installed
2. Check GPU compatibility and available memory
3. Verify all dependencies are installed in the correct order
4. For flash-attention specific issues, consult the official documentation

<a name="perf"></a>
## Performance Benchmarks

| Stage | Time (NVIDIA 4090)   | Memory Usage                                        |
| ----- |----------------------|-----------------------------------------------------|
| Data Preprocessing | ~2 min               | 4GB RAM                                             |
| NER Processing | ~30 min              | 8GB VRAM                                            | 
| LLM Selection | ~1 min per inference | 23.2GB VRAM at least for 7B, 40 Gb at least for 14B |
| Total Pipeline | ~several hours       | 24GB VRAM peak                                      |

<a name="tools"></a>
## Tools

<a name="mod"></a>
### Models
- CamemBERT-NER by Jean-Baptiste: French NER model fine-tuned for date extraction
- Qwen2.5 series by Alibaba: Advanced LLMs optimized for reasoning tasks

<a name="dat"></a>
### Data
- Datapolitics: Provided the French administrative document corpus

<a name="dev"></a>
### Device
- VLLM: High-performance LLM inference engine
- Flash Attention: Efficient attention computation

```text
Megret L., 2024. Final Project of Master 2, Automated Date Extraction Pipeline. Université Paris Cité, Master Sciences du langage - Parcours : Computational Linguistics
Domaine : Sciences humaines et sociales
```
