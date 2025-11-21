"""
Grain of Salt Labeler - CS 5342 Trust and Safety Assignment 3
==============================================================

Team Members: Andrew, Josh, Max, Minfei

OVERVIEW
--------
This labeler implements a "Grain of Salt" scoring system for political news/commentary
on Bluesky. The score ranges from 0-9, indicating how much skepticism a reader should
apply when consuming the content:

    0 = Ground truth / verified statistics (take with 0 grains of salt)
    1-2 = News with estimated stats or paraphrased information
    3-4 = News with sensational framing or urgency
    5-6 = Sensational opinion or cryptic claims
    7-8 = Highly sensational or likely exaggerated
    9 = Obviously satirical content (take with many grains of salt)

SCOPE
--------------------------------
- Focus: Political news/commentary (diverse distribution across our scale)
- NOT a misinformation labeler - we're scoring credibility/skepticism level
- This is different from fact-checking: we indicate how much "salt" to take

KEY SIGNALS
-----------
1. Account-level signals:
   - Known satire sources (The Onion, etc.)
   - Official government/news accounts
   - Parody indicators in bio/name
   
2. Content-level signals:
   - Sensational language patterns
   - Urgency markers
   - Statistical claims vs. opinions
   - First-person claims
   
3. Source credibility:
   - Verified news outlets
   - Government sources
   - Known satire/parody accounts

ITERATION NOTES
---------------
This is Version 1.0 - designed to be simple and testable.
Future iterations will add:
- LLM-based content analysis
- Historical posting pattern analysis
- News consensus checking
"""

import os
import re
import csv
from typing import List, Dict, Tuple, Optional
from atproto import Client
from pylabel.label import post_from_url, did_from_handle


class GrainOfSaltLabeler:
    """
    Automated labeler that scores political content on a 0-9 "Grain of Salt" scale.
    
    Higher scores indicate content that should be consumed with more skepticism ("grains of salt").
    """
    
    def __init__(self, client: Client, input_dir: str = None):
        """
        Initialize the Grain of Salt labeler.
        
        Args:
            client: Authenticated Bluesky ATProto client
            input_dir: Directory containing input files (optional for this labeler)
        """
        self.client = client
        self.input_dir = input_dir
        
        # Load our classification data
        self._load_known_accounts()
        self._load_signal_patterns()
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def _load_known_accounts(self):
        """
        Load lists of known account types for baseline scoring.
        
        Categories:
        - satire_accounts: Known satire/parody sources (score 9)
        - official_accounts: Government/verified news (score 0-2)
        - parody_indicators: Words in bio/name indicating parody
        """
        
        # Known satire/parody accounts (baseline score: 9)
        # These are accounts where content is obviously not meant to be taken literally
        self.satire_accounts = {
            'theonion.com',           # The Onion - famous satire
            'clickhole.com',          # ClickHole - satire
            'babylonbee.com',         # Babylon Bee - satire
            'hard-drive.net',         # Hard Drive - gaming satire
            'reductress.com',         # Reductress - satire
            'borowitz.bsky.social',   # Andy Borowitz - satire
            # Parody accounts from our dataset
            'ice-parody.bsky.social',
            'donaldjtrump-maga.bsky.social',  # Fake Trump account
            'tedcrvz.bsky.social',    # Parody Ted Cruz
        }
        
        # Official/verified accounts (baseline score: 0-2)
        # Government, established news, official sources
        self.official_accounts = {
            # Government
            'nws.noaa.gov': 0,         # National Weather Service
            'police.boston.gov': 0,    # Boston Police
            'dol.gov': 1,              # Dept of Labor
            'parks.boston.gov': 0,     # Boston Parks
            'governor.ca.gov': 2,      # CA Governor (political, some spin)
            'whitehouse.gov': 1,       # White House
            'aclu.org': 1,             # ACLU
            'crockett.house.gov': 3,   # Congressional rep (partisan)
            'pelosi.house.gov': 3,     # Congressional rep (partisan)
            
            # Major news outlets (generally lower scores)
            'nytimes.com': 1,
            'washingtonpost.com': 1,
            'bloomberg.com': 1,
            'financialtimes.com': 1,
            'theguardian.com': 2,
            'reuters.com': 0,
            'apnews.com': 0,
            'bbc.com': 1,
            'politico.com': 2,
            'npr.org': 1,
            'democracynow.org': 2,
            'thetimes.com': 1,
            'starsandstripes.bsky.social': 2,
            
            # Election/research
            'mitelectionlab.bsky.social': 1,
            'electionline.bsky.social': 2,
            'lawanddemocracy.bsky.social': 2,
            
            # Tracking/data accounts
            'trumpjet.grndcntrl.net': 0,  # Just tracking data
        }
        
        # Words that indicate parody/satire when in bio or display name
        self.parody_indicators = [
            'parody', 'satire', 'fake', 'not real', 'humor',
            'comedy', 'joke', 'unofficial', 'fan account', 'roleplay'
        ]
    
    def _load_signal_patterns(self):
        """
        Load regex patterns and word lists for content analysis.
        
        These help identify:
        - Sensational language
        - Urgency markers
        - Statistical claims
        - Opinion indicators
        """
        
        # Sensational/urgent language (increases score)
        self.sensational_words = [
            'breaking', 'urgent', 'shocking', 'explosive', 'bombshell',
            'devastating', 'horrific', 'terrifying', 'unbelievable',
            'unprecedented', 'massive', 'huge', 'incredible', 'outrageous',
            'insane', 'crazy', 'wild', 'alarming', 'critical', 'emergency'
        ]
        
        # Opinion/editorial indicators (increases score)
        self.opinion_indicators = [
            'i think', 'i believe', 'in my opinion', 'imo', 'imho',
            'clearly', 'obviously', 'of course', 'everyone knows',
            'the truth is', 'let me be clear', 'make no mistake',
            'here\'s the thing', 'hot take', 'unpopular opinion'
        ]
        
        # Sarcasm/irony markers (increases score significantly)
        self.sarcasm_markers = [
            'totally', 'definitely', 'surely', 'because that makes sense',
            '/s', '🙄', 'oh great', 'wonderful', 'fantastic',
            'what could go wrong', 'nothing to see here'
        ]
        
        # Credibility indicators (decreases score)
        self.credibility_indicators = [
            'according to', 'study shows', 'research indicates',
            'data shows', 'statistics show', 'percent', '%',
            'reported by', 'confirmed', 'verified', 'official'
        ]
        
        # All caps pattern (sensationalism indicator)
        self.caps_pattern = re.compile(r'\b[A-Z]{4,}\b')
        
        # URL pattern for checking linked sources
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
    
    # =========================================================================
    # MAIN SCORING FUNCTION
    # =========================================================================
    
    def moderate_post(self, url: str) -> List[str]:
        """
        Score a post and return the appropriate grain-of-salt label.
        
        Args:
            url: URL of the Bluesky post to analyze
            
        Returns:
            List containing single label like ["salt-5"] or empty list if error
        """
        try:
            # Fetch the post
            post = post_from_url(self.client, url)
            
            # Calculate the grain of salt score
            score, reasoning = self.calculate_score(post, url)
            
            # Return as label
            return [f"salt-{score}"]
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return []
    
    def calculate_score(self, post, url: str) -> Tuple[int, Dict]:
        """
        Calculate the grain of salt score for a post.
        
        Args:
            post: The fetched post object
            url: Original URL (for handle extraction)
            
        Returns:
            Tuple of (score: int, reasoning: dict with breakdown)
        """
        reasoning = {
            'account_score': 0,
            'content_score': 0,
            'adjustments': [],
            'final_score': 0
        }
        
        # Extract handle from URL
        handle = self._extract_handle(url)
        
        # Step 1: Get baseline score from account type
        account_score = self._score_account(handle, post)
        reasoning['account_score'] = account_score
        
        # Step 2: Analyze content
        record = self._get_record(post)
        text = record.text if record and hasattr(record, 'text') else ""
        
        content_score = self._score_content(text)
        reasoning['content_score'] = content_score
        
        # Step 3: Combine scores
        # Account score is the baseline, content can adjust it
        if account_score is not None:
            # Known account - use account score as strong baseline
            # Content can adjust by up to +/- 2
            adjustment = max(-2, min(2, content_score - 5))
            final_score = account_score + adjustment
            reasoning['adjustments'].append(f"Content adjustment: {adjustment}")
        else:
            # Unknown account - rely more on content analysis
            final_score = content_score
            reasoning['adjustments'].append("Unknown account - using content score")
        
        # Ensure score is in valid range
        final_score = max(0, min(9, final_score))
        reasoning['final_score'] = final_score
        
        return final_score, reasoning
    
    # =========================================================================
    # SCORING COMPONENTS
    # =========================================================================
    
    def _score_account(self, handle: str, post) -> Optional[int]:
        """
        Score based on account type/reputation.
        
        Args:
            handle: The account handle
            post: The post object (for profile info)
            
        Returns:
            Baseline score (0-9) or None if unknown account
        """
        handle_lower = handle.lower()
        
        # Check if it's a known satire account
        if handle_lower in self.satire_accounts:
            return 9
        
        # Check if it's a known official/news account
        if handle_lower in self.official_accounts:
            return self.official_accounts[handle_lower]
        
        # Check for parody indicators in handle
        for indicator in self.parody_indicators:
            if indicator in handle_lower:
                return 8
        
        # TODO: Future - fetch profile and check bio for parody indicators
        # TODO: Future - check account age, follower ratio, etc.
        
        return None  # Unknown account
    
    def _score_content(self, text: str) -> int:
        """
        Analyze content to determine skepticism score.
        
        Args:
            text: The post text content
            
        Returns:
            Content-based score (0-9)
        """
        if not text:
            return 5  # Default middle score for empty text
        
        text_lower = text.lower()
        score = 5  # Start at neutral
        
        # Check for sensational language
        sensational_count = sum(
            1 for word in self.sensational_words 
            if word in text_lower
        )
        score += min(sensational_count, 2)  # Cap at +2
        
        # Check for opinion indicators
        opinion_count = sum(
            1 for phrase in self.opinion_indicators 
            if phrase in text_lower
        )
        score += min(opinion_count, 2)  # Cap at +2
        
        # Check for sarcasm markers (strong signal)
        sarcasm_count = sum(
            1 for marker in self.sarcasm_markers 
            if marker in text_lower
        )
        score += min(sarcasm_count * 2, 3)  # Strong signal, cap at +3
        
        # Check for credibility indicators (reduces score)
        credibility_count = sum(
            1 for phrase in self.credibility_indicators 
            if phrase in text_lower
        )
        score -= min(credibility_count, 3)  # Cap reduction at -3
        
        # Check for excessive caps (sensationalism)
        caps_matches = self.caps_pattern.findall(text)
        if len(caps_matches) >= 2:
            score += 1
        
        # Ensure score stays in range
        return max(0, min(9, score))
    
    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    
    def _get_record(self, post):
        """Extract the record from the post object."""
        if hasattr(post, 'value'):
            return post.value
        elif hasattr(post, 'record'):
            return post.record
        return None
    
    def _extract_handle(self, url: str) -> str:
        """
        Extract handle from Bluesky post URL.
        
        Args:
            url: Post URL like https://bsky.app/profile/handle/post/xyz
            
        Returns:
            The handle string
        """
        parts = url.split("/")
        # URL format: https://bsky.app/profile/{handle}/post/{rkey}
        try:
            profile_index = parts.index('profile')
            return parts[profile_index + 1]
        except (ValueError, IndexError):
            return ""
    
    # =========================================================================
    # TESTING & EVALUATION
    # =========================================================================
    
    def evaluate_on_dataset(self, csv_path: str) -> Dict:
        """
        Evaluate labeler performance on a dataset with ground truth scores.
        
        Args:
            csv_path: Path to CSV with columns: Link, Gut check score
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'total': 0,
            'exact_matches': 0,
            'within_1': 0,
            'within_2': 0,
            'predictions': [],
            'errors': []
        }
        
        # Read the dataset
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get('Link', '').strip()
                expected_str = row.get('Gut check score', '').strip()
                
                if not url or not expected_str:
                    continue
                
                # Parse expected score (handle "9+" as 9)
                try:
                    expected = int(expected_str.replace('+', ''))
                except ValueError:
                    continue
                
                results['total'] += 1
                
                # Get prediction
                labels = self.moderate_post(url)
                if labels:
                    predicted = int(labels[0].replace('salt-', ''))
                else:
                    predicted = 5  # Default
                
                # Record result
                diff = abs(predicted - expected)
                results['predictions'].append({
                    'url': url,
                    'expected': expected,
                    'predicted': predicted,
                    'diff': diff
                })
                
                if diff == 0:
                    results['exact_matches'] += 1
                if diff <= 1:
                    results['within_1'] += 1
                if diff <= 2:
                    results['within_2'] += 1
        
        # Calculate metrics
        if results['total'] > 0:
            results['accuracy_exact'] = results['exact_matches'] / results['total']
            results['accuracy_within_1'] = results['within_1'] / results['total']
            results['accuracy_within_2'] = results['within_2'] / results['total']
        
        return results
    
    def print_evaluation_report(self, results: Dict):
        """Print a formatted evaluation report."""
        print("\n" + "="*60)
        print("GRAIN OF SALT LABELER - EVALUATION REPORT")
        print("="*60)
        print(f"\nTotal posts evaluated: {results['total']}")
        print(f"\nAccuracy Metrics:")
        print(f"  Exact matches:  {results['exact_matches']}/{results['total']} "
              f"({results.get('accuracy_exact', 0)*100:.1f}%)")
        print(f"  Within ±1:      {results['within_1']}/{results['total']} "
              f"({results.get('accuracy_within_1', 0)*100:.1f}%)")
        print(f"  Within ±2:      {results['within_2']}/{results['total']} "
              f"({results.get('accuracy_within_2', 0)*100:.1f}%)")
        
        # Show worst predictions
        print("\nLargest discrepancies:")
        sorted_preds = sorted(results['predictions'], 
                             key=lambda x: x['diff'], reverse=True)
        for pred in sorted_preds[:5]:
            print(f"  Expected {pred['expected']}, got {pred['predicted']} "
                  f"(diff={pred['diff']})")
            print(f"    {pred['url'][:60]}...")
        
        print("\n" + "="*60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function for testing the Grain of Salt labeler.
    """
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv(override=True)
    USERNAME = os.getenv("USERNAME")
    PW = os.getenv("PW")
    
    parser = argparse.ArgumentParser(
        description="Grain of Salt Labeler for Political Content"
    )
    parser.add_argument(
        "input_csv", 
        type=str,
        help="CSV file with post URLs and expected scores"
    )
    parser.add_argument(
        "--single-url",
        type=str,
        help="Test a single URL instead of CSV"
    )
    args = parser.parse_args()
    
    # Login to Bluesky
    client = Client()
    client.login(USERNAME, PW)
    
    # Create labeler
    labeler = GrainOfSaltLabeler(client)
    
    if args.single_url:
        # Test single URL
        labels = labeler.moderate_post(args.single_url)
        print(f"Result: {labels}")
    else:
        # Evaluate on dataset
        results = labeler.evaluate_on_dataset(args.input_csv)
        labeler.print_evaluation_report(results)


if __name__ == "__main__":
    main()
