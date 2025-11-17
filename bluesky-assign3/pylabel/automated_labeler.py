"""Implementation of automated moderator"""

import csv
import re
from typing import List, Set
from .label import post_from_url
from atproto import Client
import imagehash
from PIL import Image
import requests
from io import BytesIO

T_AND_S_LABEL = "t-and-s"
DOG_LABEL = "dog"
THRESH = 0.25

class AutomatedLabeler:
    """Automated labeler implementation"""

    def __init__(self, client: Client, input_dir):
        self.client = client
        self.input_dir = input_dir
        
        # Load T&S words (case-insensitive)
        self.ts_words = self._load_ts_words()
        
        # Load T&S domains
        self.ts_domains = self._load_ts_domains()

        # Load news domains
        self.news_domains = self._load_news_domains()
        
        # Load dog image hashes
        self.dog_hashes = self._load_dog_hashes()
    
    def _get_record(self, post):
        """Extract the record from the post object"""
        if hasattr(post, 'value'):
            return post.value
        elif hasattr(post, 'record'):
            return post.record
        return None

    def _load_ts_words(self) -> Set[str]:
        """Load T&S words from CSV, convert to lowercase"""
        words = set()
        with open(f"{self.input_dir}/t-and-s-words.csv", 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                words.add(row['Word'].lower())
        return words
    
    def _load_ts_domains(self) -> Set[str]:
        """Load T&S domains from CSV"""
        domains = set()
        with open(f"{self.input_dir}/t-and-s-domains.csv", 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                domain = row['Domain'].strip().strip('/').lower()
                # Remove http:// or https://
                domain = domain.replace('https://', '').replace('http://', '')
                domains.add(domain)
        return domains
    
    def moderate_post(self, url: str) -> List[str]:
        """Apply moderation to the post specified by the given url"""
        labels = []
        
        # Fetch the post
        post = post_from_url(self.client, url)
        record = self._get_record(post)
        
        # Check for T&S words/domains
        if self._contains_ts_content(post):
            labels.append(T_AND_S_LABEL)
        
        # Check for news sources
        news_labels = self._check_news_sources(post)
        labels.extend(news_labels)
        
        # Check for dog images
        if self._contains_dog_image(post):
            labels.append(DOG_LABEL)
        
        return labels

    def _contains_ts_content(self, post) -> bool:
        """Check if post contains T&S words or domains"""
        record = self._get_record(post)
        if not record:
            return False
        
        # Check text for T&S words (case-insensitive)
        if hasattr(record, 'text') and record.text:
            text_lower = record.text.lower()
            
            # Check for T&S words
            for word in self.ts_words:
                if word in text_lower:
                    return True
            
            # Also check for T&S domains in the text itself (fallback for non-faceted URLs)
            for domain in self.ts_domains:
                # Remove www. prefix for comparison
                domain_clean = domain.replace('www.', '').replace('https://', '').replace('http://', '').strip('/')
                if domain_clean in text_lower:
                    return True
        
        
        # Check for links with T&S domains in facets
        if hasattr(record, 'facets') and record.facets:
            for facet in record.facets:
                for feature in facet.features:
                    if hasattr(feature, 'uri'):
                        domain = self._extract_domain(feature.uri)
                        # Compare with and without www. prefix
                        if domain and (domain.lower() in self.ts_domains or 
                                    f'www.{domain.lower()}' in self.ts_domains):
                            return True
                                
        return False

    
    def _load_news_domains(self) -> dict:
        """Load news domains and their labels"""
        news_map = {}
        with open(f"{self.input_dir}/news-domains.csv", 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                news_map[row['Domain'].lower()] = row['Source']
        return news_map
    
    def _check_news_sources(self, post) -> List[str]:
        """Check post for news source links and return appropriate labels"""
        labels = set()  # Use set to avoid duplicate labels
        record = self._get_record(post)
        
        if not record:
            return []
        
        # Check facets first
        if hasattr(record, 'facets') and record.facets:
            for facet in record.facets:
                for feature in facet.features:
                    if hasattr(feature, 'uri'):
                        domain = self._extract_domain(feature.uri)
                        if domain and domain.lower() in self.news_domains:
                            labels.add(self.news_domains[domain.lower()])
        
        # Fallback: check for news domains in plain text
        if hasattr(record, 'text') and record.text:
            text_lower = record.text.lower()
            for domain, source in self.news_domains.items():
                # Check if domain appears in text
                if domain in text_lower:
                    labels.add(source)
        
        return list(labels)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove 'www.' prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    
    def _load_dog_hashes(self) -> List[str]:
        """Load dog images and compute their perceptual hashes"""
        import os
        dog_hashes = []
        dog_dir = f"{self.input_dir}/dog-list-images"
        
        for filename in os.listdir(dog_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(dog_dir, filename)
                img = Image.open(img_path)
                phash = imagehash.phash(img)
                dog_hashes.append(phash)
        
        return dog_hashes

    def _contains_dog_image(self, post) -> bool:
        """Check if post contains an image matching the dog list"""
        record = self._get_record(post)
        
        if not record:
            return False
        
        if not hasattr(self, 'dog_hashes') or not self.dog_hashes:
            return False
        
        # Get the DID from the post URI
        did = None
        if hasattr(post, 'uri'):
            uri_parts = post.uri.split('/')
            if len(uri_parts) >= 3:
                did = uri_parts[2]
            
        # Check if post has embedded images
        if hasattr(record, 'embed') and record.embed:
            embed = record.embed
            
            # Handle different embed types
            images = []
            if hasattr(embed, 'images'):
                images = embed.images
            elif hasattr(embed, 'image'):
                images = [embed.image]
            
            for image in images:
                img_url = None
                
                if hasattr(image, 'image'):
                    img_blob = image.image
                    
                    if hasattr(img_blob, 'ref') and hasattr(img_blob.ref, 'link'):
                        cid = img_blob.ref.link
                        
                        if did and cid:
                            img_url = f"https://cdn.bsky.app/img/feed_fullsize/plain/{did}/{cid}@jpeg"
                
                if not img_url:
                    continue
                
                try:
                    response = requests.get(img_url, timeout=10)
                    img = Image.open(BytesIO(response.content))
                    post_hash = imagehash.phash(img)
                    
                    # Find the minimum distance to any dog hash
                    min_distance = min(post_hash - dog_hash for dog_hash in self.dog_hashes)
                    threshold_bits = THRESH * 64
                    
                    
                    if min_distance <= threshold_bits:
                        return True
                        
                except Exception as e:
                    print(f"Error processing image: {e}")
                    continue
        
        return False