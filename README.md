# Watermarkhub

This repository contains the code for the Paper: ***WatermarkHub: A Robust and Automated Watermark Agent for Online Data Platform***.

<!-- Due to time constraints, this code submission was made hastily, but the core code is included. **We assure that we will update the code and documentation with more user-friendly versions as soon as possible.** -->

## 1 Requirements

We recommended the following dependencies.

- python==3.10.10
- langchain==0.2.9
- langchain-openai==0.1.17
- pydantic>=2.0.0

For more recommended dependencies, please refer to the file  [`requirements.txt`](https://anonymous.4open.science/r/DiversityMark-5503/requirements.txt).

``` bash
conda create -n watermark python=3.10.10 -y
conda activate watermark
pip install -r requirements.txt
```

## 2 How to use

### 2.1 Generating parse code

We use the SL.log dataset as an example.

Before generating the code, set up your api key in `.env`.

```python
API_KEY = ""
BASE_URL = ""
```

Run `chain4code_gen.py` to obtain the parse code: 

```bash
python chain4code_gen.py
```


### 2.2 Watermarking and Detecting

Run `parse_data.py` to process and watermark specified dataset : 

```bash
python parse_data.py
```

## 3 Verification Module 


### 3.1 Overview

The verification module provides three core functionalities:

1. **Schema Validation**: Validates that watermarked data conforms to the expected schema structure
2. **Constraint Checking**: Verifies both local constraints (per-record changes) and global constraints (overall modification rate)
3. **Signature Verification**: Confirms the watermark signature can be correctly extracted and matched


```bash
python verification_enhanced.py

```
