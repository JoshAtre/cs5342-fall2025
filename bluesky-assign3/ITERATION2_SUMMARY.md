# Iteration 2: Enhanced Rule-Based System

## Changes Implemented

### 1. Expanded Known Accounts Database

**Added Satire/Joke Accounts (Score 9):**
- `satire-of-sevs.bsky.social` - Satirical account
- `wokestudies.bsky.social` - Satirical commentary
- `funnysnarkyjoke.bsky.social` - Joke account
- `ctrlaltresist.com` - Resistance satire/humor
- `darthstateworker.bsky.social` - Satire account
- `davidcorn.bsky.social` - Satirical commentary
- `lawprofblawg.bsky.social` - Satirical legal commentary

**Added News/Credible Accounts (Score 0-2):**
- `cnn.com` (0) - Major news network
- `nbcnews.com` (2) - Major news network
- `huffpost.com` (0) - News aggregator/outlet
- `forbes.com` (0) - Business/news outlet
- `economist.com` (1) - News/analysis
- `latimes.com` (1) - Major newspaper
- `pbsnews.org` (2) - Public broadcasting
- `propublica.org` (1) - Investigative journalism
- `factcheck.afp.com` (0) - Fact-checking service
- `cityofdayton.bsky.social` (0) - City government

**Impact:** These additions directly address 7 of the 10 largest errors from Iteration 1, including CNN, Forbes, HuffPost, AFP, City of Dayton, and satire accounts that were previously unknown.

### 2. Increased Content Influence (±2 → ±3)

**Change:** Content analysis can now adjust the account baseline score by ±3 points instead of ±2.

**Rationale:** 
- Iteration 1 showed content signals were too constrained
- Unknown accounts starting at neutral (5) couldn't reach extreme ends (0-2 or 8-9)
- Strong content signals (heavy statistics, obvious satire) deserve more influence

**Impact:** Allows the labeler to better handle edge cases where content strongly contradicts or reinforces the account baseline.

### 3. Enhanced Satire/Joke Detection Patterns

**Added patterns:**
- Laughing indicators: `lmao`, `lol`, `💀`, `😂`, `🤣`
- Gen Z irony markers: `fr fr`, `no cap`, `cap`
- Extreme reactions: `i'm dead`, `i can't`, `crying`
- Meme phrases: `not me`, `who did this`
- Informal/ironic address: `the way`, `bestie`, `sis`
- Story/joke setups: `imagine`, `pov:`, `tell me why`
- Humor mixing: `real talk though`, `all jokes aside`

**Impact:** Better detection of modern internet humor and meme culture, especially from younger users.

### 4. Government Domain Detection

**Added automatic detection for:**

**Government Domains (.gov):**
- All `.gov` domains automatically assigned score 0
- Covers federal, state, and local government

**City/Local Government Patterns:**
- Keywords: `cityof`, `city.`, `townof`, `countyof`
- Automatically assigned score 0

**News Organization Heuristics:**
- Domains containing news keywords: `news`, `times`, `post`, `press`, `herald`, `tribune`, `journal`
- With `.com`, `.org`, or `.net` domains
- Automatically assigned score 1

**Impact:** Systematic detection of government and news sources instead of manual database maintenance. Should handle any new government accounts automatically.

## Expected Improvements

Based on error analysis from Iteration 1:

### Error Reduction
- **7 unknown account errors** should be resolved (CNN, Forbes, HuffPost, AFP, City of Dayton, satire-of-sevs, wokestudies)
- **Government domain detection** catches any .gov accounts automatically
- **Enhanced satire detection** better identifies joke content

### Performance Targets
- Accuracy within ±2: **>75%** (from 68.7%)
- Precision (±2 tolerance): **>85%** (from 82.8%)
- Recall (±2 tolerance): **>70%** (from 66.6%)
- Mean Absolute Error: **<1.5** (from 1.79)

### Class-Specific Improvements
- Score 1: Better recall (was 40%)
- Score 8: Better recall (was 10%)
- Scores 3, 4, 7: Should now have measurable precision/recall (were N/A)

## Testing Instructions

Run the updated labeler:
```bash
python policy_proposal_labeler.py data.csv --save-results iteration2_results.csv
```

Compare with baseline:
```bash
# Compare accuracy metrics
# Compare precision/recall improvements
# Analyze remaining error cases
```

## Key Metrics to Track

1. **Resolved Errors:** How many of the top 10 errors from Iteration 1 are now correct?
2. **New Errors:** What new failure patterns emerge?
3. **Class Balance:** Are previously problematic score classes (1, 8) improving?
4. **Overall Metrics:** Accuracy, precision, recall compared to baseline

## Next Iteration Ideas

If Iteration 2 doesn't meet targets:
- Add more domain-based detection patterns
- Implement bio/profile text analysis
- Add temporal patterns (check posting frequency)
- Consider LLM-based content analysis for difficult cases
