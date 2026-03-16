"""
Scrape Grokipedia articles. Target: 1 GB of clean text.
FAST: 150 workers, connection pooling, single-fetch link discovery.
"""
import requests
import concurrent.futures
import re
import os
import time
import json
from html.parser import HTMLParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'training_data')
PROGRESS_FILE = os.path.join(BASE_DIR, 'scrape_progress.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

WORKERS = 150
BATCH_SIZE = 500
HEADERS = {'User-Agent': 'BinaryAI-Research/1.0 (non-commercial AI training research)'}

# Persistent session for connection pooling
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
adapter = requests.adapters.HTTPAdapter(
    pool_connections=150,
    pool_maxsize=150,
    max_retries=1,
)
SESSION.mount('https://', adapter)
SESSION.mount('http://', adapter)

SEED_ARTICLES = [
    "Earth", "Sun", "Moon", "Water", "Human", "United_States", "World_War_II",
    "Mathematics", "Physics", "Chemistry", "Biology", "Computer", "Internet",
    "History", "Philosophy", "Science", "Language", "Music", "Art", "Religion",
    "Jesus", "Muhammad", "Buddha", "Democracy", "Economics", "Law", "Medicine",
    "Albert_Einstein", "Isaac_Newton", "Charles_Darwin", "William_Shakespeare",
    "Leonardo_da_Vinci", "Napoleon", "Alexander_the_Great", "Julius_Caesar",
    "Abraham_Lincoln", "Martin_Luther_King_Jr.", "Mahatma_Gandhi", "Nelson_Mandela",
    "China", "India", "Japan", "Germany", "France", "United_Kingdom", "Russia",
    "Brazil", "Australia", "Canada", "Mexico", "Egypt", "South_Africa", "Nigeria",
    "Pacific_Ocean", "Atlantic_Ocean", "Amazon_River", "Nile", "Mount_Everest",
    "Solar_System", "Milky_Way", "Big_Bang", "Black_hole", "DNA", "Evolution",
    "Gravity", "Light", "Electricity", "Atom", "Cell_(biology)", "Brain",
    "Heart", "Climate_change", "Artificial_intelligence", "Quantum_mechanics",
    "World_War_I", "Cold_War", "Roman_Empire", "Ancient_Greece", "Ancient_Egypt",
    "Renaissance", "Industrial_Revolution", "French_Revolution", "American_Revolution",
    "Constitution", "Human_rights", "United_Nations", "European_Union", "NATO",
    "Python_(programming_language)", "JavaScript", "HTML", "CSS", "Linux",
    "Google", "Apple_Inc.", "Microsoft", "Amazon_(company)", "Tesla,_Inc.",
    "Bitcoin", "Cryptocurrency", "Machine_learning", "Neural_network",
    "Love", "Death", "Time", "Space", "Energy", "Matter", "Force",
    "Football", "Basketball", "Baseball", "Soccer", "Olympics",
    "Novel", "Poetry", "Film", "Television", "Photography",
    "Cooking", "Agriculture", "Architecture", "Engineering", "Aviation",
    "Dog", "Cat", "Horse", "Elephant", "Whale", "Lion", "Eagle",
    "Tree", "Flower", "Ocean", "Mountain", "Desert", "Forest", "River",
    "Psychology", "Sociology", "Anthropology", "Archaeology", "Astronomy",
    "Calculus", "Algebra", "Geometry", "Statistics", "Logic",
    "Oxygen", "Carbon", "Iron", "Gold", "Silver", "Uranium", "Hydrogen",
    "Volcano", "Earthquake", "Hurricane", "Tornado", "Tsunami",
    "Vaccine", "Antibiotic", "Surgery", "Cancer", "Diabetes",
    "Protein", "Vitamin", "Mineral", "Nutrition", "Exercise",
    "Marriage", "Family", "Education", "University", "School",
    "Money", "Bank", "Stock_market", "Inflation", "Trade",
    "War", "Peace", "Diplomacy", "Terrorism", "Nuclear_weapon",
    "Planet", "Star", "Galaxy", "Comet", "Asteroid",
    "Continent", "Island", "Peninsula", "Glacier", "Coral_reef",
    "Mythology", "Folklore", "Legend", "Fairy_tale", "Epic_poetry",
]


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'nav', 'header', 'footer', 'aside'):
            self.skip = True
        if tag in ('p', 'h1', 'h2', 'h3', 'h4', 'li'):
            self.text.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'nav', 'header', 'footer', 'aside'):
            self.skip = False
        if tag in ('p', 'h1', 'h2', 'h3', 'h4', 'li', 'br'):
            self.text.append('\n')

    def handle_data(self, data):
        if not self.skip:
            self.text.append(data)

    def get_text(self):
        return ''.join(self.text)


def clean_text(html):
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(html)
    except:
        return ""
    text = extractor.get_text()
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()
    if len(text) < 200:
        return ""
    return text


def discover_links(html):
    links = re.findall(r'href="/page/([^"]+)"', html)
    return list(set(l for l in links if ':' not in l and not l.startswith('Special') and not l.startswith('Talk')))


def fetch_article(title):
    """Fetch article - returns (text, links, title) in ONE request."""
    url = f"https://grokipedia.com/page/{title}"
    try:
        resp = SESSION.get(url, timeout=10)
        if resp.status_code != 200:
            return None, [], title
        raw = resp.text
        text = clean_text(raw)
        links = discover_links(raw) if text and len(text) > 500 else []
        return text, links, title
    except Exception:
        return None, [], title


def scrape():
    target_bytes = 1 * 1024 * 1024 * 1024  # 1 GB
    seen = set()
    queue = list(SEED_ARTICLES)
    batch_num = 0

    # Load progress
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            prog = json.load(f)
            seen = set(prog.get('seen', []))
            saved_queue = prog.get('queue', [])
            if saved_queue:
                queue = saved_queue
            print(f"Resuming: {prog.get('total_articles', 0)} articles seen")

    # Count actual bytes from files on disk (single source of truth)
    total_bytes = 0
    total_articles = 0
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.txt'):
            total_bytes += os.path.getsize(os.path.join(OUTPUT_DIR, f))
            total_articles += 1

    print(f"Target: 1 GB | Current: {total_bytes/1024/1024:.0f} MB ({total_articles} articles)")
    print(f"Queue: {len(queue)} | Workers: {WORKERS}")
    print()

    # Persistent thread pool - no recreation overhead
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS)

    while total_bytes < target_bytes and queue:
        batch_num += 1
        batch = []
        while queue and len(batch) < BATCH_SIZE:
            title = queue.pop(0)
            if title not in seen:
                seen.add(title)
                batch.append(title)

        if not batch:
            break

        # Fire all requests at once
        futures = {pool.submit(fetch_article, t): t for t in batch}
        new_links = []

        for future in concurrent.futures.as_completed(futures):
            text, links, title = future.result()
            if text and len(text) > 500:
                safe_name = re.sub(r'[^\w\-]', '_', title)[:80]
                filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                total_bytes += len(text)
                total_articles += 1
                new_links.extend(links)

        # Add discovered links to queue
        for link in new_links:
            if link not in seen:
                queue.append(link)

        mb = total_bytes / 1024 / 1024
        gb = total_bytes / 1024 / 1024 / 1024
        pct = total_bytes / target_bytes * 100
        print(f"  Batch {batch_num}: {total_articles} articles | "
              f"{mb:.0f} MB ({gb:.2f} GB) | {pct:.1f}% of 1 GB | "
              f"queue: {len(queue)}")

        # Save progress (including queue for resume)
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({
                'seen': list(seen)[-10000:],
                'total_bytes': total_bytes,
                'total_articles': total_articles,
                'queue': queue[:5000],
            }, f)

    pool.shutdown(wait=False)
    print()
    print(f"Done: {total_articles} articles | {total_bytes/1024/1024/1024:.2f} GB")


if __name__ == '__main__':
    print("=" * 55)
    print("  Grokipedia Scraper - Target: 1 GB")
    print("=" * 55)
    print()
    scrape()
