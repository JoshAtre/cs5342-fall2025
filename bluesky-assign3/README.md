# CS 5342 Assignment 3 - Grain of Salt Labeler

**Group Number:** 15

**Team Members:**
- Josh Atre
- Andrew Beketov
- Matthias Corkran
- Minfei Shen

---

## Project Overview

This project implements a "Grain of Salt" content labeler for political posts on Bluesky. The labeler assigns scores from 0-9 indicating how much skepticism a reader should apply when consuming content:

- **0-2:** Ground truth, verified facts, credible news
- **3-4:** News with sensational framing or some spin
- **5-6:** Sensational opinion or cryptic claims
- **7-8:** Highly sensational or likely exaggerated
- **9:** Obviously satirical content

---

## Files Submitted

### Core Implementation Files

1. **`policy_proposal_labeler.py`** (Main Implementation)
   - Complete implementation of the Grain of Salt labeler
   - Uses rule-based scoring combining account reputation and content analysis
   - Includes comprehensive evaluation framework with accuracy, precision, and recall metrics
   - Supports both single-URL testing and full dataset evaluation

### Data Files

2. **`data.csv`** (Labeled Dataset)
   - 150 manually labeled political posts from Bluesky
   - Columns: Link (post URL), Gut check score (0-9), Notes
   - Used for training, testing, and evaluation

3. **`baseline_results.csv`** (Evaluation Results)
   - Detailed results from running the labeler on the dataset
   - Columns: url, expected, predicted, diff, time_seconds, correct
   - Shows per-post predictions and timing information

### Supporting Files

4. **`.env-TEMPLATE`** (Environment Configuration Template)
   - Template for required environment variables
   - Copy to `.env` and fill in your Bluesky credentials

5. **`pylabel/`** (Directory)
   - Helper functions for interacting with Bluesky API
   - Includes `label.py` with `post_from_url()` and `did_from_handle()` functions

### Documentation

6. **`README.md`** (This File)
   - Project overview and setup instructions
   - Complete guide for running the labeler and tests

---

## Prerequisites

### Required Software
- Python 3.8 or higher
- pip (Python package manager)

### Required Python Packages
```bash
pip install atproto python-dotenv
```

### Bluesky Account
You will need a Bluesky account with credentials to run the labeler. The account is used to fetch posts via the AT Protocol API.

---

## Setup Instructions

### 1. Install Dependencies

```bash
# Install required Python packages
pip install atproto python-dotenv
```

### 2. Configure Environment Variables

```bash
# Copy the template
cp .env-TEMPLATE .env

# Edit .env and add your Bluesky credentials
# Example .env content:
# USERNAME=your_bluesky_handle.bsky.social
# PW=your_app_password
```

**Note:** For security, use an app-specific password rather than your main account password. You can generate app passwords in your Bluesky account settings.

### 3. Verify Setup

```bash
# Test that the environment is configured correctly
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Setup OK' if os.getenv('USERNAME') else 'Please configure .env')"
```

---

## Running the Labeler

### Full Dataset Evaluation

To run the labeler on the complete dataset and generate evaluation metrics:

```bash
python policy_proposal_labeler.py data.csv --save-results baseline_results.csv
```

**Expected Output:**
```
======================================================================
GRAIN OF SALT LABELER - COMPREHENSIVE EVALUATION REPORT
======================================================================

📊 DATASET STATISTICS
──────────────────────────────────────────────────────────────────────
Total posts evaluated: 150

🎯 ACCURACY METRICS
──────────────────────────────────────────────────────────────────────
  Exact matches:        X/150 ( XX.X%)
  Within ±1:            X/150 ( XX.X%)
  Within ±2:            X/150 ( XX.X%)
  Mean Absolute Error: X.XX

📈 PRECISION & RECALL (±2 tolerance)
──────────────────────────────────────────────────────────────────────
  NOTE: Metrics calculated using ±2 point tolerance
  Macro-averaged Precision: XX.X%
  Macro-averaged Recall:    XX.X%

[... additional metrics and error analysis ...]
```

**Processing Time:** Approximately XX-XX seconds for 150 posts (~XXXms per post)

### Testing a Single URL

To test the labeler on a specific post with detailed reasoning:

```bash
python policy_proposal_labeler.py data.csv --single-url "https://bsky.app/profile/theonion.com/post/3m4tjuvb5yp2c"
```

**Example Output:**
```
Testing URL: https://bsky.app/profile/theonion.com/post/3m4tjuvb5yp2c

Predicted Score: 9
Account baseline: 9
Content score: 7
Adjustments: ['Content adjustment: 0']
```

---

## Understanding the Output

### Evaluation Report Sections

1. **Dataset Statistics**
   - Total number of posts processed

2. **Accuracy Metrics**
   - Exact matches: Predictions that exactly match the expected score
   - Within ±1: Predictions within 1 point of expected (useful tolerance)
   - Within ±2: Predictions within 2 points of expected (practical tolerance)
   - Mean Absolute Error (MAE): Average distance from expected scores

3. **Precision & Recall**
   - Calculated using ±2 tolerance (predictions within 2 points considered correct)
   - Macro-averaged across all score classes (0-9)
   - Important for understanding class-specific performance

4. **Per-Class Metrics**
   - Breakdown of precision, recall, and F1-score for each score (0-9)
   - Support: Number of posts with each expected score
   - Identifies which score ranges perform best/worst

5. **Performance Metrics**
   - Processing time and throughput
   - Memory usage
   - Network requests made

6. **Error Analysis**
   - Top 10 largest discrepancies between predicted and expected scores
   - Includes URLs for investigating failure cases

### Results CSV Format

The `baseline_results.csv` file contains:
- **url:** The Bluesky post URL
- **expected:** The manually labeled score (0-9)
- **predicted:** The labeler's predicted score (0-9)
- **diff:** Absolute difference between expected and predicted
- **time_seconds:** Time taken to process this post
- **correct:** "yes" if exact match, "no" otherwise

---

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'atproto'`
```bash
Solution: pip install atproto python-dotenv
```

**Issue:** `KeyError: 'USERNAME'` or authentication errors
```bash
Solution: Ensure .env file exists and contains valid credentials
# Check with: cat .env
```

**Issue:** `FileNotFoundError: data.csv`
```bash
Solution: Run from the correct directory containing data.csv
# Or provide full path: python policy_proposal_labeler.py /path/to/data.csv
```

**Issue:** Rate limiting or network errors
```bash
Solution: The labeler includes automatic rate limiting and retries
# If persistent, check your network connection and Bluesky service status
```

**Issue:** Slow processing
```bash
Expected: ~277ms per post, ~40 seconds for 150 posts
If slower: Check network latency, ensure you're not hitting rate limits
```
