import os
import re
import csv
import time
import tracemalloc
from typing import List, Dict, Tuple, Optional
from atproto import Client
from pylabel.label import post_from_url, did_from_handle


class GrainOfSaltLabeler:
    """
    Automated labeler that scores political content on a 0-9 "Grain of Salt" scale.
    
    Higher scores indicate content that should be consumed with more skepticism.
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
    
    # DATA LOADING
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
            # Established satire outlets
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
            
            # V2.0: Additional satire/joke accounts from dataset analysis
            'satire-of-sevs.bsky.social',     # Satire account
            'wokestudies.bsky.social',        # Satire account
            'funnysnarkyjoke.bsky.social',    # Joke account
            'ctrlaltresist.com',              # Satire/resistance humor
            'darthstateworker.bsky.social',   # Satire account
            'davidcorn.bsky.social',          # Often satirical commentary
            'lawprofblawg.bsky.social',       # Satirical legal commentary
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
            
            # V2.0: Additional city/local government from dataset
            'cityofdayton.bsky.social': 0,   # City government
            
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
            
            # V2.0: Additional news outlets from dataset analysis
            'cnn.com': 0,              # Major news network
            'nbcnews.com': 2,          # Major news network
            'huffpost.com': 0,         # News aggregator/outlet
            'forbes.com': 0,           # Business/news outlet
            'economist.com': 1,        # News/analysis
            'latimes.com': 1,          # Major newspaper
            'pbsnews.org': 2,          # Public broadcasting
            'propublica.org': 1,       # Investigative journalism
            'factcheck.afp.com': 0,    # Fact-checking service
            
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
            'what could go wrong', 'nothing to see here',
            
            # V2.0: Enhanced satire/joke detection
            'lmao', 'lol', '💀', '😂', '🤣',  # Laughing indicators
            'fr fr', 'no cap', 'cap',  # Gen Z irony markers
            'i\'m dead', 'i can\'t', 'crying',  # Extreme reactions
            'not me', 'who did this',  # Meme phrases
            'the way', 'bestie', 'sis',  # Informal/ironic address
            'imagine', 'pov:', 'tell me why',  # Story/joke setups
            'real talk though', 'all jokes aside',  # Signals mixing humor with seriousness
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
    
    # MAIN SCORING FUNCTION
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
            # V2.0: Content can adjust by up to +/- 3 (increased from 2)
            adjustment = max(-3, min(3, content_score - 5))
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
    
    # SCORING COMPONENTS    
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
        
        # V2.0: Check for government domains (.gov)
        if '.gov' in handle_lower:
            # Federal/state/local government domains are generally authoritative
            return 0
        
        # V2.0: Check for city/local government patterns
        if any(keyword in handle_lower for keyword in ['cityof', 'city.', 'townof', 'countyof']):
            # Local government accounts
            return 0
        
        # V2.0: Check for major news organization domains
        news_domains = ['.com', '.org', '.net']
        major_news_keywords = ['news', 'times', 'post', 'press', 'herald', 'tribune', 'journal']
        if any(domain in handle_lower for domain in news_domains):
            if any(keyword in handle_lower for keyword in major_news_keywords):
                return 1  # Likely a news organization
        
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
    
    # HELPER FUNCTIONS    
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
    
    # TESTING & EVALUATION    
    def evaluate_on_dataset(self, csv_path: str) -> Dict:
        """
        Comprehensive evaluation with accuracy, precision, recall, and performance metrics.
        
        Args:
            csv_path: Path to CSV with columns: Link, Gut check score
            
        Returns:
            Dictionary with evaluation metrics including:
            - Accuracy (exact, within ±1, within ±2)
            - Precision & Recall (macro-averaged and per-class)
            - Performance (time, memory, throughput)
            - Confusion matrix
            - Error analysis
        """
        # Start performance tracking
        start_time = time.time()
        tracemalloc.start()
        
        results = {
            'total': 0,
            'exact_matches': 0,
            'within_1': 0,
            'within_2': 0,
            'predictions': [],
            'confusion_matrix': [[0]*10 for _ in range(10)],  # 10x10 for scores 0-9
            'performance': {
                'total_time': 0,
                'avg_time_per_post': 0,
                'peak_memory_mb': 0,
                'network_requests': 0
            }
        }
        
        # Read the dataset
        post_times = []
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
                results['performance']['network_requests'] += 1  # Each post requires a fetch
                
                # Time individual post processing
                post_start = time.time()
                labels = self.moderate_post(url)
                predicted = int(labels[0].replace('salt-', '')) if labels else 5
                post_time = time.time() - post_start
                post_times.append(post_time)
                
                # Update confusion matrix
                results['confusion_matrix'][expected][predicted] += 1
                
                # Record result
                diff = abs(predicted - expected)
                results['predictions'].append({
                    'url': url,
                    'expected': expected,
                    'predicted': predicted,
                    'diff': diff,
                    'time_seconds': post_time
                })
                
                # Update accuracy metrics
                if diff == 0:
                    results['exact_matches'] += 1
                if diff <= 1:
                    results['within_1'] += 1
                if diff <= 2:
                    results['within_2'] += 1
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results['performance']['total_time'] = total_time
        results['performance']['avg_time_per_post'] = sum(post_times) / len(post_times) if post_times else 0
        results['performance']['peak_memory_mb'] = peak / 1024 / 1024
        
        # Calculate overall metrics
        if results['total'] > 0:
            results['accuracy_exact'] = results['exact_matches'] / results['total']
            results['accuracy_within_1'] = results['within_1'] / results['total']
            results['accuracy_within_2'] = results['within_2'] / results['total']
            results['mae'] = sum(p['diff'] for p in results['predictions']) / results['total']
        
        # Calculate per-class precision, recall, F1
        results['class_metrics'] = self._calculate_class_metrics(results['confusion_matrix'])
        
        # Calculate macro-averaged metrics
        precisions = [m['precision'] for m in results['class_metrics'].values() if m['precision'] is not None]
        recalls = [m['recall'] for m in results['class_metrics'].values() if m['recall'] is not None]
        
        results['macro_precision'] = sum(precisions) / len(precisions) if precisions else 0
        results['macro_recall'] = sum(recalls) / len(recalls) if recalls else 0
        
        return results
    
    def _calculate_class_metrics(self, confusion_matrix: List[List[int]]) -> Dict:
        """
        Calculate precision, recall, and F1 for each score class.
        
        NOTE: Using ±2 tolerance for precision/recall calculations.
        A prediction is considered correct if it's within 2 points of the true label.
        
        Args:
            confusion_matrix: 10x10 confusion matrix
            
        Returns:
            Dictionary mapping score -> {precision, recall, f1, support}
        """
        class_metrics = {}
        
        for class_idx in range(10):
            # True positives: predictions within ±2 of the true class
            tp = sum(confusion_matrix[class_idx][pred] 
                    for pred in range(10) 
                    if abs(pred - class_idx) <= 2)
            
            # False positives: predictions for this class that were actually other classes (outside ±2)
            fp = sum(confusion_matrix[true][class_idx] 
                    for true in range(10) 
                    if abs(true - class_idx) > 2)
            
            # False negatives: actual instances of this class that were predicted outside ±2
            fn = sum(confusion_matrix[class_idx][pred] 
                    for pred in range(10) 
                    if abs(pred - class_idx) > 2)
            
            # Support: total instances of this class
            support = sum(confusion_matrix[class_idx])
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else None
            recall = tp / (tp + fn) if (tp + fn) > 0 else None
            
            if precision and recall and (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = None
            
            class_metrics[class_idx] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        return class_metrics
    
    def print_evaluation_report(self, results: Dict):
        """Print comprehensive evaluation report with all metrics."""
        print("\n" + "="*70)
        print("GRAIN OF SALT LABELER - COMPREHENSIVE EVALUATION REPORT")
        print("="*70)
        
        # Dataset statistics
        print(f"\n DATASET STATISTICS")
        print(f"{'─'*70}")
        print(f"Total posts evaluated: {results['total']}")
        
        # Accuracy metrics
        print(f"\n ACCURACY METRICS")
        print(f"{'─'*70}")
        print(f"  Exact matches:       {results['exact_matches']:>3}/{results['total']:<3} "
              f"({results.get('accuracy_exact', 0)*100:>5.1f}%)")
        print(f"  Within ±1:           {results['within_1']:>3}/{results['total']:<3} "
              f"({results.get('accuracy_within_1', 0)*100:>5.1f}%)")
        print(f"  Within ±2:           {results['within_2']:>3}/{results['total']:<3} "
              f"({results.get('accuracy_within_2', 0)*100:>5.1f}%)")
        print(f"  Mean Absolute Error: {results.get('mae', 0):.2f}")
        
        # Precision & Recall
        print(f"\n PRECISION & RECALL (±2 tolerance)")
        print(f"{'─'*70}")
        print(f"  NOTE: Metrics calculated using ±2 point tolerance")
        print(f"  Macro-averaged Precision: {results.get('macro_precision', 0)*100:.1f}%")
        print(f"  Macro-averaged Recall:    {results.get('macro_recall', 0)*100:.1f}%")
        
        # Per-class metrics
        print(f"\n PER-CLASS METRICS")
        print(f"{'─'*70}")
        print(f"  Score | Support | Precision | Recall | F1-Score")
        print(f"  {'-'*54}")
        
        for score in range(10):
            metrics = results['class_metrics'].get(score, {})
            support = metrics.get('support', 0)
            
            if support > 0:
                prec = metrics.get('precision')
                rec = metrics.get('recall')
                f1 = metrics.get('f1')
                
                prec_str = f"{prec*100:>5.1f}%" if prec else "  N/A"
                rec_str = f"{rec*100:>5.1f}%" if rec else "  N/A"
                f1_str = f"{f1:.3f}" if f1 else " N/A"
                
                print(f"    {score}   |   {support:>3}   | {prec_str:>8} | {rec_str:>6} | {f1_str:>5}")
        
        # Performance metrics
        print(f"\n PERFORMANCE METRICS")
        print(f"{'─'*70}")
        perf = results['performance']
        print(f"  Total processing time:    {perf['total_time']:.2f} seconds")
        print(f"  Average time per post:    {perf['avg_time_per_post']*1000:.0f} ms")
        print(f"  Peak memory usage:        {perf['peak_memory_mb']:.2f} MB")
        print(f"  Network requests:         {perf['network_requests']}")
        print(f"  Throughput:               {results['total']/perf['total_time']:.1f} posts/second")
        
        # Error analysis
        print(f"\n ERROR ANALYSIS - Top 10 Largest Discrepancies")
        print(f"{'─'*70}")
        sorted_preds = sorted(results['predictions'], key=lambda x: x['diff'], reverse=True)
        
        for i, pred in enumerate(sorted_preds[:10], 1):
            print(f"  {i:>2}. Expected {pred['expected']}, got {pred['predicted']} "
                  f"(diff={pred['diff']}, {pred['time_seconds']*1000:.0f}ms)")
            print(f"      {pred['url'][:65]}...")
        
        print("\n" + "="*70 + "\n")
    
    def save_detailed_results(self, results: Dict, output_path: str):
        """Save detailed results to CSV for further analysis."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'url', 'expected', 'predicted', 'diff', 'time_seconds', 'correct'
            ])
            writer.writeheader()
            
            for pred in results['predictions']:
                writer.writerow({
                    'url': pred['url'],
                    'expected': pred['expected'],
                    'predicted': pred['predicted'],
                    'diff': pred['diff'],
                    'time_seconds': pred['time_seconds'],
                    'correct': 'yes' if pred['diff'] == 0 else 'no'
                })
        
        print(f"Detailed results saved to: {output_path}")


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
        help="Test a single URL with detailed reasoning"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        help="Save detailed results to CSV file"
    )
    args = parser.parse_args()
    
    print("Logging in to Bluesky...")
    client = Client()
    client.login(USERNAME, PW)
    
    labeler = GrainOfSaltLabeler(client)
    print("Labeler initialized.\n")
    
    if args.single_url:
        # Test single URL with reasoning
        print(f"Testing URL: {args.single_url}")
        post = post_from_url(client, args.single_url)
        score, reasoning = labeler.calculate_score(post, args.single_url)
        
        print(f"\nPredicted Score: {score}")
        print(f"Account baseline: {reasoning['account_score']}")
        print(f"Content score: {reasoning['content_score']}")
        print(f"Adjustments: {reasoning['adjustments']}")
    else:
        # Evaluate on dataset
        print(f"Evaluating on: {args.input_csv}\n")
        results = labeler.evaluate_on_dataset(args.input_csv)
        labeler.print_evaluation_report(results)
        
        if args.save_results:
            labeler.save_detailed_results(results, args.save_results)


if __name__ == "__main__":
    main()
