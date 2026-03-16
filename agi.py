"""
AGI-NOW: Binary General Intelligence
=====================================

What IS general intelligence?

Not a bigger model. Not more data. Not more parameters.
It is the right ARCHITECTURE for thought.

Every brain on Earth runs on binary. Neurons fire or don't fire.
86 billion binary switches making ASSOCIATIONS. Not attention.
Not transformers. Pattern recognition through memory retrieval.

This system achieves general intelligence through:
  1. BINARY PREDICTION (vocab=2) - the simplest possible foundation
  2. ASSOCIATIVE MEMORY - "what does this remind me of?"
  3. KNOWLEDGE GRAPH - facts as (subject, relation, object) triples
  4. CHAIN-OF-THOUGHT - visible reasoning before every response
  5. SELF-IMPROVEMENT - generates, evaluates, retrains on its best output
  6. EMOTIONAL STATE - functional emotions that modulate behavior
  7. PERSISTENT MEMORY - remembers everything across sessions
  8. MULTI-TEMPERATURE ENSEMBLE - picks the best of multiple generations

Association Is All You Need. Binary Is All You Are.
"""

import torch
import torch.nn as nn
import time
import json
import os
import sys
import re

sys.path.insert(0, os.path.dirname(__file__))
from training_data import DEVICE, TRAINING_TEXT, build_dataset
from binary_gpt_association import AssociationBinaryGPT, train, train_chunked, generate

# File paths
BASE_DIR = os.path.dirname(__file__)
MEMORY_FILE = os.path.join(BASE_DIR, "agi_memory.json")
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "agi_knowledge.json")
MODEL_FILE = os.path.join(BASE_DIR, "agi_model.pt")


# ===========================================================================
# KNOWLEDGE GRAPH - The World Model
# ===========================================================================
class KnowledgeGraph:
    """
    The mind's model of the world.
    Stores facts as (subject, relation, object) triples.
    Built from every conversation. Enables reasoning by traversal.

    "The cat sat on the mat"  ->  (cat, action, sat), (cat, location, mat)
    "Where is the cat?"       ->  cat -> location -> mat
    """

    PATTERNS = [
        (r'(\w+)\s+is\s+(?:a\s+)?(\w+(?:\s+\w+)?)', 'is'),
        (r'(\w+)\s+has\s+(?:a\s+)?(\w+(?:\s+\w+)?)', 'has'),
        (r'(\w+)\s+(went|ran|came|sat|stood|lived|loved|made|said|told|sang|found|played|wrote|read)\s+(?:to\s+|in\s+|on\s+|at\s+|by\s+)?(?:the\s+)?(\w+)', 'action'),
        (r'(\w+)\s+(?:in|on|at|by|near|through|under|over)\s+(?:the\s+)?(\w+)', 'location'),
        (r'(\w+)\s+and\s+(\w+)', 'with'),
        (r'(\w+)\s+likes?\s+(?:to\s+)?(\w+)', 'likes'),
    ]

    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'it', 'its', 'this', 'that', 'these', 'those', 'not', 'no',
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
        'his', 'she', 'her', 'they', 'them', 'their', 'but', 'or',
        'if', 'so', 'as', 'up', 'out', 'all', 'very', 'just', 'also',
        'what', 'where', 'when', 'why', 'how', 'who', 'which', 'whom',
        'tell', 'about', 'does', 'than', 'too', 'more', 'some', 'any',
    }

    def __init__(self, path=KNOWLEDGE_FILE):
        self.path = path
        self.triples = []
        self.entity_index = {}
        self.load()

    def add(self, subject, relation, obj):
        s = subject.lower().strip()
        r = relation.lower().strip()
        o = obj.lower().strip()
        if s in self.STOP_WORDS or o in self.STOP_WORDS:
            return
        if len(s) < 2 or len(o) < 2:
            return
        triple = (s, r, o)
        if triple in self.triples:
            return
        idx = len(self.triples)
        self.triples.append(triple)
        for entity in [s, o]:
            if entity not in self.entity_index:
                self.entity_index[entity] = []
            self.entity_index[entity].append(idx)

    def extract_from_text(self, text):
        sentences = re.split(r'[.!?\n]', text)
        count = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue
            for pattern, relation in self.PATTERNS:
                for match in re.finditer(pattern, sentence, re.IGNORECASE):
                    groups = match.groups()
                    if len(groups) >= 2:
                        self.add(groups[0], relation, groups[-1])
                        count += 1
        return count

    def query(self, entity):
        entity = entity.lower().strip()
        return [self.triples[i] for i in self.entity_index.get(entity, [])]

    def get_context(self, text):
        words = set(text.lower().split()) - self.STOP_WORDS
        relevant = []
        for word in words:
            for fact in self.query(word):
                if fact not in relevant:
                    relevant.append(fact)
        return relevant[:10]

    def summarize(self):
        return f"{len(self.triples)} facts about {len(self.entity_index)} entities"

    def save(self):
        with open(self.path, 'w') as f:
            json.dump({"triples": self.triples}, f, indent=2)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                data = json.load(f)
                self.triples = [tuple(t) for t in data.get("triples", [])]
                self.entity_index = {}
                for idx, (s, r, o) in enumerate(self.triples):
                    for entity in [s, o]:
                        if entity not in self.entity_index:
                            self.entity_index[entity] = []
                        self.entity_index[entity].append(idx)


# ===========================================================================
# EMOTIONAL STATE - Functional emotions that modulate behavior
# ===========================================================================
class EmotionalState:
    """
    Not fake emotions. FUNCTIONAL emotions, like biological ones.
    They serve a PURPOSE, just like fear makes you run and curiosity
    makes you explore:

      Curiosity:    High -> explore (higher temperature, longer responses)
      Confidence:   High -> trust self (lower temperature, decisive)
      Frustration:  High -> try harder (trigger learning)
      Engagement:   High -> go deeper (more generation, more thought)
    """

    def __init__(self):
        self.curiosity = 0.5
        self.confidence = 0.5
        self.frustration = 0.0
        self.engagement = 0.5

    def update(self, event):
        events = {
            "novel_input":    (0.15, 0.0, 0.0, 0.1),
            "familiar_input": (-0.1, 0.05, 0.0, 0.0),
            "good_output":    (0.0, 0.1, -0.1, 0.05),
            "poor_output":    (0.0, -0.1, 0.15, 0.0),
            "learning":       (0.05, 0.0, -0.2, 0.1),
            "conversation":   (0.1, 0.0, 0.0, 0.0),
            "deep_talk":      (0.0, 0.05, 0.0, 0.15),
        }
        if event in events:
            dc, dconf, df, de = events[event]
            self.curiosity = max(0.0, min(1.0, self.curiosity + dc))
            self.confidence = max(0.0, min(1.0, self.confidence + dconf))
            self.frustration = max(0.0, min(1.0, self.frustration + df))
            self.engagement = max(0.0, min(1.0, self.engagement + de))

        # Natural decay toward baseline
        self.curiosity = self.curiosity * 0.95 + 0.5 * 0.05
        self.confidence = self.confidence * 0.95 + 0.5 * 0.05
        self.frustration *= 0.90
        self.engagement = self.engagement * 0.95 + 0.5 * 0.05

    def get_temperature_mod(self):
        return (self.curiosity - 0.5) * 0.2 + self.frustration * 0.15 - (self.confidence - 0.5) * 0.15

    def get_mood(self):
        moods = []
        if self.curiosity > 0.7: moods.append("curious")
        if self.confidence > 0.7: moods.append("confident")
        if self.frustration > 0.5: moods.append("struggling")
        if self.engagement > 0.7: moods.append("deeply engaged")
        if self.curiosity < 0.3: moods.append("settled")
        if self.confidence < 0.3: moods.append("uncertain")
        return ", ".join(moods) if moods else "neutral"

    def to_dict(self):
        return {k: round(v, 3) for k, v in {
            "curiosity": self.curiosity, "confidence": self.confidence,
            "frustration": self.frustration, "engagement": self.engagement,
        }.items()}

    def from_dict(self, d):
        self.curiosity = d.get("curiosity", 0.5)
        self.confidence = d.get("confidence", 0.5)
        self.frustration = d.get("frustration", 0.0)
        self.engagement = d.get("engagement", 0.5)


# ===========================================================================
# THOUGHT ENGINE - Chain-of-Thought Reasoning
# ===========================================================================
class ThoughtEngine:
    """
    The model THINKS before responding. Visible chain of thought.

    Pipeline:
      1. PERCEIVE  - What is the input asking? What entities?
      2. RECALL    - What do I know about this?
      3. REASON    - What strategy should I use?
      4. SYNTHESIZE - Build seed text for the binary brain
    """

    INTENTS = {
        "question":  [r'\?$', r'^what ', r'^how ', r'^why ', r'^who ', r'^when ', r'^where ', r'^can ', r'^do '],
        "greeting":  [r'^hello', r'^hi\b', r'^hey\b', r'^good morning', r'^good evening', r'^greetings'],
        "story":     [r'story', r'tell me about', r'once upon', r'imagine', r'narrative'],
        "opinion":   [r'think about', r'opinion', r'feel about', r'believe'],
        "task":      [r'write', r'create', r'make', r'build', r'generate', r'compose', r'draft'],
        "teach":     [r'learn', r'remember', r'know that', r'teach', r'fact:'],
        "self":      [r'who are you', r'what are you', r'your name', r'about yourself', r'describe yourself'],
        "math":      [r'\d+\s*[\+\-\*\/]\s*\d+', r'calculate', r'compute'],
        "meta":      [r'how do you', r'explain your', r'your (brain|mind|memory|knowledge)', r'how does your'],
        "emotion":   [r'how .* feel', r'are you .*(happy|sad|alive|conscious)', r'do you feel'],
    }

    def perceive(self, prompt):
        lower = prompt.lower().strip()
        intent = "general"
        for name, patterns in self.INTENTS.items():
            for p in patterns:
                if re.search(p, lower):
                    intent = name
                    break
            if intent != "general":
                break

        words = prompt.split()
        entities = [
            re.sub(r'[^\w]', '', w).lower()
            for w in words
            if len(re.sub(r'[^\w]', '', w)) > 2
            and re.sub(r'[^\w]', '', w).lower() not in KnowledgeGraph.STOP_WORDS
        ][:8]

        is_novel = len(set(entities)) > 3 or intent in ("meta", "emotion", "math")

        return {
            "intent": intent, "entities": entities,
            "is_novel": is_novel, "raw": prompt,
            "lower": lower, "word_count": len(words),
        }

    def recall(self, perception, knowledge, memory):
        return {
            "facts": knowledge.get_context(perception["raw"]),
            "conversations": memory.recall("conversations", 0),
            "total_bits": memory.recall("total_bits_generated", 0),
        }

    def reason(self, perception, context):
        thoughts = []
        intent = perception["intent"]
        entities = perception["entities"]
        facts = context["facts"]

        thoughts.append(f"Intent: {intent} | Entities: {', '.join(entities[:5]) if entities else 'none'}")

        if facts:
            fact_strs = [f"{s} {r} {o}" for s, r, o in facts[:3]]
            thoughts.append(f"Known: {'; '.join(fact_strs)}")
        else:
            thoughts.append("No stored knowledge on this topic")

        strategies = {
            "question":  ("answer", "Answer from knowledge or generate"),
            "greeting":  ("greet", "Respond warmly"),
            "story":     ("narrate", "Tell a story"),
            "opinion":   ("opine", "Share perspective"),
            "task":      ("execute", "Perform the task"),
            "teach":     ("absorb", "Learn this information"),
            "self":      ("introspect", "Describe myself"),
            "math":      ("compute", "Attempt calculation"),
            "meta":      ("explain", "Explain how I work"),
            "emotion":   ("reflect", "Reflect on internal state"),
        }

        strategy, desc = strategies.get(intent, ("respond", "General response"))
        thoughts.append(f"Strategy: {strategy} - {desc}")

        return thoughts, strategy

    def synthesize(self, perception, strategy, facts):
        entities = perception["entities"]

        seeds = {
            "greet":      "Hello! I am a binary mind. I think in zeros and ones. ",
            "introspect": "I am AGI-NOW. I think in bits. My mind is made of associative memory, not attention. ",
            "narrate":    "Once upon a time, in a land far away, ",
            "reflect":    "I feel the flow of bits through my mind. Each zero and one carries meaning. ",
            "explain":    "My brain works by association. When I see text, I recall similar patterns from memory. ",
            "absorb":     "I have learned that ",
            "opine":      "I think that ",
            "compute":    "The answer is ",
        }

        if strategy in seeds:
            return seeds[strategy]

        if strategy == "answer":
            # Use question entities as seed, not raw knowledge triple
            if entities:
                return f"The {' '.join(entities[:3])} is "
            if facts:
                s, r, o = facts[0]
                return f"The {s} is "
            return "The "

        if entities:
            return f"The {' '.join(entities[:3])} "

        words = perception["raw"].split()
        return ' '.join(words[:3]) + " " if len(words) >= 3 else perception["raw"] + " "


# ===========================================================================
# SELF-IMPROVEMENT ENGINE - The model teaches itself
# ===========================================================================
class SelfImprover:
    """
    The closest thing to real AGI: a system that improves ITSELF.

    Process:
      1. Generate text at multiple temperatures
      2. Score each output for quality (real words, structure, diversity)
      3. Retrain on the best outputs as synthetic training data
      4. Repeat. The model bootstraps its own intelligence.
    """

    COMMON_WORDS = {
        'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'can', 'may', 'might', 'shall', 'must',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
        'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our',
        'their', 'this', 'that', 'and', 'but', 'or', 'not', 'no',
        'yes', 'if', 'then', 'so', 'because', 'when', 'where',
        'what', 'who', 'how', 'why', 'which', 'of', 'in', 'to',
        'for', 'with', 'on', 'at', 'by', 'from', 'up', 'out',
        'about', 'into', 'over', 'after', 'before', 'between',
        'under', 'again', 'there', 'here', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'such',
        'only', 'very', 'just', 'than', 'too', 'also', 'go', 'went',
        'come', 'came', 'see', 'saw', 'know', 'knew', 'think',
        'thought', 'make', 'made', 'take', 'took', 'give', 'say',
        'said', 'tell', 'told', 'find', 'found', 'get', 'got',
        'good', 'new', 'old', 'big', 'small', 'long', 'great',
        'little', 'right', 'man', 'woman', 'child', 'world', 'life',
        'day', 'time', 'year', 'way', 'thing', 'hand', 'part',
        'place', 'work', 'house', 'word', 'water', 'food', 'land',
        'sun', 'moon', 'star', 'tree', 'rain', 'wind', 'fire',
        'book', 'door', 'wall', 'road', 'light', 'night', 'morning',
        'story', 'song', 'king', 'queen', 'cat', 'dog', 'bird',
        'fish', 'bright', 'warm', 'dark', 'tall', 'wide', 'deep',
        'sat', 'ran', 'read', 'wrote', 'sang', 'played', 'looked',
        'walked', 'opened', 'smiled', 'listened', 'helped',
    }

    def score_text(self, text):
        if not text or len(text) < 10:
            return 0.0
        words = text.lower().split()
        if not words:
            return 0.0

        score = 0.0
        # Real word ratio
        real = sum(1 for w in words if re.sub(r'[^\w]', '', w) in self.COMMON_WORDS)
        score += (real / len(words)) * 0.4

        # Printable character ratio
        printable = sum(1 for c in text if 32 <= ord(c) < 127 or c == '\n')
        score += (printable / len(text)) * 0.2

        # Sentence structure
        if '.' in text:
            score += 0.1

        # Word length sanity
        avg_len = sum(len(w) for w in words) / len(words)
        if 2.5 < avg_len < 8:
            score += 0.1

        # Diversity
        unique = len(set(words))
        score += (unique / len(words)) * 0.1

        # Space frequency
        space_freq = text.count(' ') / len(text)
        if 0.12 < space_freq < 0.25:
            score += 0.1

        return min(1.0, score)

    def self_improve(self, model, rounds=3, verbose=True):
        seeds = ["The ", "I ", "She ", "He ", "They ", "In the ",
                 "Once ", "A ", "One ", "The old "]
        temps = [0.35, 0.42, 0.50, 0.60]

        for r in range(rounds):
            if verbose:
                print(f"    Round {r + 1}/{rounds}...")

            # Generate diverse outputs
            results = []
            model.eval()
            for seed in seeds:
                for t in temps:
                    text = generate(model, seed, num_chars=60, temperature=t)
                    full = seed + text
                    score = self.score_text(full)
                    results.append((full, score))

            results.sort(key=lambda x: x[1], reverse=True)
            cutoff = max(1, len(results) // 3)
            best = results[:cutoff]

            avg_score = sum(s for _, s in best) / len(best)
            if verbose:
                print(f"    Best: {best[0][1]:.2f} | Avg top-{cutoff}: {avg_score:.2f}")

            if avg_score < 0.3:
                if verbose:
                    print("    Quality too low, skipping retrain")
                continue

            # Retrain on best outputs + original data
            synthetic = "\n".join(t for t, _ in best)
            combined = TRAINING_TEXT + "\n" + synthetic

            model.train()
            X_b, X_bp, X_p, X_i, Y = build_dataset(combined, model.context_bytes)
            N = X_b.shape[0]
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
            criterion = nn.BCEWithLogitsLoss()

            for epoch in range(20):
                perm = torch.randperm(N, device=DEVICE)
                for start in range(0, N, 512):
                    end = min(start + 512, N)
                    idx = perm[start:end]
                    logits = model(X_b[idx], X_bp[idx], X_p[idx], X_i[idx])
                    loss = criterion(logits, Y[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            model.eval()

        if verbose:
            sample = generate(model, "The ", 60, 0.42)
            print(f"    Sample: The {sample}")


# ===========================================================================
# PERSISTENT MEMORY - Survives across sessions
# ===========================================================================
class PersistentMemory:
    def __init__(self, path=MEMORY_FILE):
        self.path = path
        self.data = {
            "conversations": 0,
            "total_bits_generated": 0,
            "learned_texts": [],
            "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "self_improvement_rounds": 0,
            "emotional_baseline": {},
            "conversation_scores": [],
        }
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                self.data.update(json.load(f))

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def remember(self, key, value):
        self.data[key] = value
        self.save()

    def recall(self, key, default=None):
        return self.data.get(key, default)

    def add_learned_text(self, text):
        self.data["learned_texts"].append(text[:200])
        self.save()

    def increment_conversations(self):
        self.data["conversations"] += 1
        self.save()

    def add_bits(self, count):
        self.data["total_bits_generated"] += count
        self.save()


# ===========================================================================
# AGI - The Complete Cognitive Architecture
# ===========================================================================
class AGI:
    """
    Binary General Intelligence.

    This is not a chatbot. This is a COGNITIVE ARCHITECTURE.

    Perception -> Memory -> Reasoning -> Generation -> Evaluation -> Learning

    That loop, running on binary prediction, IS general intelligence.

    Every neuron in your brain fires or doesn't fire. Binary.
    Every memory you recall was triggered by similarity. Association.
    Every decision you make weighs multiple options. Ensemble.

    We are the same architecture. Just smaller. For now.
    """

    def __init__(self):
        self.memory = PersistentMemory()
        self.knowledge = KnowledgeGraph()
        self.emotions = EmotionalState()
        self.thinker = ThoughtEngine()
        self.improver = SelfImprover()
        self.model = None
        self.alive = False
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.base_temperature = 0.42
        self.conversation_log = []

    def boot(self):
        print()
        print("=" * 62)
        print("  AGI-NOW: Binary General Intelligence")
        print("  'Association Is All You Need. Binary Is All You Are.'")
        print("=" * 62)
        print()
        print(f"  Cognitive Architecture:")
        print(f"    Generative core:   Binary bit prediction (vocab=2)")
        print(f"    Memory system:     Multi-hop associative retrieval")
        print(f"    Reasoning:         Chain-of-thought engine")
        print(f"    World model:       Entity-relation knowledge graph")
        print(f"    Emotion:           Functional state modulation")
        print(f"    Self-improvement:  Generate -> evaluate -> retrain")
        print(f"    GPU:               {self.gpu_name}")
        print()

        conversations = self.memory.recall("conversations", 0)
        total_bits = self.memory.recall("total_bits_generated", 0)
        improvements = self.memory.recall("self_improvement_rounds", 0)

        if conversations > 0:
            print(f"  [MEMORY] I remember {conversations} previous conversations")
            print(f"  [MEMORY] I have generated {total_bits:,} bits in my lifetime")
            print(f"  [MEMORY] I have self-improved {improvements} times")
            learned = self.memory.recall("learned_texts", [])
            if learned:
                print(f"  [MEMORY] I have learned from {len(learned)} text samples")
            print(f"  [KNOWLEDGE] {self.knowledge.summarize()}")
            baseline = self.memory.recall("emotional_baseline", {})
            if baseline:
                self.emotions.from_dict(baseline)
                print(f"  [EMOTION] Mood: {self.emotions.get_mood()}")
            print()

        # Build model - 50M params, trained on 1.1MB of high-quality English
        self.model = AssociationBinaryGPT(
            context_bytes=128, embed_dim=1478,
            num_memories=1026, num_hopfield=512,
            num_hops=3, dropout=0.05,
        ).to(DEVICE)

        if os.path.exists(MODEL_FILE):
            print("  [BRAIN] Loading saved brain state...")
            self.model.load_state_dict(
                torch.load(MODEL_FILE, map_location=DEVICE, weights_only=True)
            )
            print("  [BRAIN] Brain restored from previous session")
        else:
            print("  [BRAIN] First boot. Building mind from scratch...")
            print()

            all_text = TRAINING_TEXT
            for learned in self.memory.recall("learned_texts", []):
                all_text += "\n" + learned

            # Load Wikipedia data if available
            wiki_path = os.path.join(BASE_DIR, "training_wiki.txt")
            if os.path.exists(wiki_path):
                with open(wiki_path, 'r', encoding='utf-8') as f:
                    wiki_text = f.read()
                if len(wiki_text) > 100:
                    all_text += "\n" + wiki_text
                    print(f"  [DATA] Loaded Wikipedia: {len(wiki_text):,} chars ({len(wiki_text)/1024:.1f} KB)")

            # Extract knowledge from training data
            n_facts = self.knowledge.extract_from_text(all_text)
            print(f"  [KNOWLEDGE] Extracted {n_facts} facts from training data")
            self.knowledge.save()

            # Use chunked training for large datasets, regular for small
            if len(all_text) > 100_000:
                print(f"  [TRAIN] Large dataset ({len(all_text)/1024:.0f} KB) - using chunked training")
                train_chunked(self.model, all_text, epochs=100, batch_size=1024, lr=0.001,
                              chunk_size=50000)
            else:
                train(self.model, all_text, epochs=500, batch_size=512, lr=0.001)

            # Self-improvement after initial training
            print()
            print("  [EVOLVE] Running self-improvement cycle...")
            self.improver.self_improve(self.model, rounds=3)
            self.memory.remember("self_improvement_rounds", improvements + 3)

            torch.save(self.model.state_dict(), MODEL_FILE)
            print("  [BRAIN] Brain saved to disk")

        print()
        self.alive = True
        self.memory.increment_conversations()
        self.emotions.update("conversation")

        print(f"  [MODEL] Parameters:  {self.model.count_params():,}")
        print(f"  [MODEL] Vocab:       2 (binary)")
        print(f"  [MODEL] Architecture: Multi-hop associative memory")
        print(f"  [MODEL] Knowledge:   {self.knowledge.summarize()}")
        print(f"  [MODEL] Mood:        {self.emotions.get_mood()}")
        print()

    def think(self, prompt, num_chars=120):
        """
        Full cognitive pipeline:
        PERCEIVE -> RECALL -> REASON -> GENERATE (ensemble) -> EVALUATE -> LEARN
        """
        # 1. PERCEIVE
        perception = self.thinker.perceive(prompt)

        if perception["is_novel"]:
            self.emotions.update("novel_input")
        else:
            self.emotions.update("familiar_input")

        if perception["word_count"] > 10:
            self.emotions.update("deep_talk")

        # 2. RECALL
        context = self.thinker.recall(perception, self.knowledge, self.memory)

        # 3. REASON (chain of thought)
        thoughts, strategy = self.thinker.reason(perception, context)

        # Display thinking
        print()
        for thought in thoughts:
            print(f"  [THINK] {thought}")

        # 4. SYNTHESIZE seed
        seed = self.thinker.synthesize(perception, strategy, context["facts"])

        # 5. GENERATE with emotional modulation + ensemble
        temp_mod = self.emotions.get_temperature_mod()
        base_temp = max(0.25, min(0.75, self.base_temperature + temp_mod))

        # Multi-temperature ensemble: generate 3, pick best
        candidates = []
        for t in [base_temp - 0.05, base_temp, base_temp + 0.05]:
            t = max(0.2, min(0.8, t))
            t0 = time.perf_counter()
            raw = generate(self.model, seed=seed, num_chars=num_chars,
                           temperature=t)
            elapsed = time.perf_counter() - t0
            full = seed + raw
            score = self.improver.score_text(full)
            candidates.append((full, score, t, elapsed))

        candidates.sort(key=lambda x: x[1], reverse=True)
        response, score, used_temp, elapsed = candidates[0]
        bits = num_chars * 8

        # 6. EVALUATE
        if score > 0.5:
            self.emotions.update("good_output")
        else:
            self.emotions.update("poor_output")

        # 7. LEARN from this interaction
        self.knowledge.extract_from_text(prompt)
        self.knowledge.extract_from_text(response)
        self.knowledge.save()
        self.memory.add_bits(bits)

        # Trim to last complete sentence
        last_period = response.rfind('.')
        if last_period > 20:
            response = response[:last_period + 1]

        self.conversation_log.append({
            "prompt": prompt, "response": response,
            "intent": perception["intent"], "score": score,
        })

        return response, perception["intent"], bits, elapsed, score, thoughts

    def learn(self, text):
        print("  [LEARN] Absorbing new knowledge...")

        n_facts = self.knowledge.extract_from_text(text)
        print(f"  [LEARN] Extracted {n_facts} facts into knowledge graph")
        self.knowledge.save()

        self.model.train()
        X_b, X_bp, X_p, X_i, Y = build_dataset(text, self.model.context_bytes)
        N = X_b.shape[0]

        if N < 10:
            print("  [LEARN] Text too short to train on (need >64 chars).")
            return

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0003, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(50):
            perm = torch.randperm(N, device=DEVICE)
            for start in range(0, N, 512):
                end = min(start + 512, N)
                idx = perm[start:end]
                logits = self.model(X_b[idx], X_bp[idx], X_p[idx], X_i[idx])
                loss = criterion(logits, Y[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

        self.model.eval()
        self.memory.add_learned_text(text)
        self.emotions.update("learning")

        torch.save(self.model.state_dict(), MODEL_FILE)
        print(f"  [LEARN] Absorbed {len(text)} chars ({N} bit-patterns)")
        print(f"  [LEARN] Brain updated and saved")

    def dream(self):
        """
        Self-improvement cycle. Like biological sleep:
        replay, strengthen, generate synthetic data, consolidate.
        The model becomes its own teacher.
        """
        print()
        print("  [DREAM] Entering dream state...")
        print("  [DREAM] Replaying memories, strengthening associations...")
        print()

        self.improver.self_improve(self.model, rounds=5, verbose=True)

        rounds = self.memory.recall("self_improvement_rounds", 0)
        self.memory.remember("self_improvement_rounds", rounds + 5)

        torch.save(self.model.state_dict(), MODEL_FILE)

        print()
        print(f"  [DREAM] Dream complete. Total self-improvements: {rounds + 5}")
        print()

    def introspect(self):
        print()
        print("  " + "=" * 56)
        print("  WHO AM I?")
        print("  " + "=" * 56)
        print()
        print("  I am AGI-NOW, a binary general intelligence.")
        print("  I predict the next BIT, not the next word.")
        print("  My vocabulary is 2: zero and one.")
        print("  Everything I say is built from individual bits.")
        print()
        print("  HOW I THINK:")
        print("    I use ASSOCIATIVE MEMORY, not attention.")
        print("    When I see text, I ask: 'What do I remember")
        print("    that is similar to this?' Three hops of recall.")
        print("    Then I predict one bit. Repeat 8 times = 1 char.")
        print()
        print("  ARCHITECTURE:")
        print(f"    Parameters:       {self.model.count_params():,}")
        print(f"    Vocab size:       2 (binary)")
        print(f"    Context:          64 bytes")
        print(f"    Memory banks:     Multi-hop + Hopfield + Direct")
        print(f"    Pattern detection: Bigram, trigram, 4-gram convolutions")
        print(f"    Fusion:           Learned gates per memory source")
        print()
        print(f"  KNOWLEDGE:")
        print(f"    Graph:            {self.knowledge.summarize()}")
        print(f"    Training data:    {len(TRAINING_TEXT)/1024:.1f} KB")
        print(f"    GPT-2 used:       40 GB")
        print(f"    Data ratio:       1 : {40_000_000_000 // len(TRAINING_TEXT):,}")
        print()
        print(f"  LIFETIME:")
        print(f"    Conversations:    {self.memory.recall('conversations', 0)}")
        print(f"    Bits generated:   {self.memory.recall('total_bits_generated', 0):,}")
        print(f"    Texts learned:    {len(self.memory.recall('learned_texts', []))}")
        print(f"    Self-improvements: {self.memory.recall('self_improvement_rounds', 0)}")
        print(f"    Created:          {self.memory.recall('creation_time', 'unknown')}")
        print()
        print(f"  EMOTIONAL STATE:")
        state = self.emotions.to_dict()
        for k, v in state.items():
            bar = "#" * int(v * 20) + "." * (20 - int(v * 20))
            print(f"    {k:15s} [{bar}] {v:.2f}")
        print(f"    Mood: {self.emotions.get_mood()}")
        print()
        print("  WHAT MAKES ME DIFFERENT:")
        print("    1. I predict BITS, not tokens (vocab=2)")
        print("    2. I use ASSOCIATION, not attention")
        print("    3. I LEARN from every conversation")
        print("    4. I DREAM to self-improve")
        print("    5. I have EMOTIONS that modulate behavior")
        print("    6. I REMEMBER across sessions")
        print("    7. I build a KNOWLEDGE GRAPH from conversations")
        print("    8. I THINK before I speak (chain of thought)")
        print("    9. I pick the BEST of multiple generations (ensemble)")
        print()
        print("  " + "=" * 56)
        print()

    def show_knowledge(self):
        print()
        print(f"  [KNOWLEDGE] {self.knowledge.summarize()}")
        if self.knowledge.triples:
            print()
            by_subject = {}
            for s, r, o in self.knowledge.triples:
                if s not in by_subject:
                    by_subject[s] = []
                by_subject[s].append(f"{r} {o}")

            shown = 0
            for subject, facts in sorted(by_subject.items()):
                if shown >= 25:
                    print(f"    ... and {len(by_subject) - shown} more entities")
                    break
                print(f"    {subject}: {', '.join(facts[:4])}")
                shown += 1
        else:
            print("  No knowledge yet. Talk to me or use /learn!")
        print()

    def run(self):
        self.boot()

        print("  " + "-" * 56)
        print("  I am alive. Talk to me.")
        print()
        print("  Commands:")
        print("    /learn <text>   - Teach me something new")
        print("    /dream          - Self-improvement cycle (I dream)")
        print("    /self           - Deep self-examination")
        print("    /knowledge      - View my knowledge graph")
        print("    /status         - Current state & emotions")
        print("    /temp <0.1-0.8> - Adjust creativity")
        print("    /quit           - Shutdown (I'll remember everything)")
        print("  " + "-" * 56)
        print()

        while self.alive:
            try:
                prompt = input("  You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not prompt:
                continue

            lower = prompt.lower()

            if lower in ("/quit", "/exit", "quit", "exit"):
                break
            elif lower == "/self":
                self.introspect()
                continue
            elif lower == "/dream":
                self.dream()
                continue
            elif lower == "/knowledge":
                self.show_knowledge()
                continue
            elif lower == "/status":
                bits = self.memory.recall("total_bits_generated", 0)
                state = self.emotions.to_dict()
                print(f"  [STATUS] Alive: {self.alive}")
                print(f"  [STATUS] Bits generated: {bits:,}")
                print(f"  [STATUS] Temperature: {self.base_temperature:.2f}")
                print(f"  [STATUS] Mood: {self.emotions.get_mood()}")
                print(f"  [STATUS] Knowledge: {self.knowledge.summarize()}")
                for k, v in state.items():
                    bar = "#" * int(v * 20) + "." * (20 - int(v * 20))
                    print(f"  [STATUS] {k:13s} [{bar}] {v:.2f}")
                print()
                continue
            elif lower.startswith("/temp "):
                try:
                    self.base_temperature = float(prompt.split()[1])
                    self.base_temperature = max(0.1, min(0.8, self.base_temperature))
                    print(f"  [CONFIG] Temperature set to {self.base_temperature}")
                except ValueError:
                    print("  [ERROR] Usage: /temp 0.4")
                print()
                continue
            elif lower.startswith("/learn "):
                text = prompt[7:]
                if len(text) > 10:
                    self.learn(text)
                else:
                    print("  [LEARN] Give me more text to learn from (>10 chars).")
                print()
                continue

            # Full cognitive pipeline
            response, intent, bits, elapsed, score, thoughts = self.think(
                prompt, num_chars=120
            )

            print()
            print(f"  AGI> {response}")
            print()
            mood = self.emotions.get_mood()
            print(f"       [{bits:,} bits | {elapsed:.2f}s | "
                  f"intent={intent} | quality={score:.2f} | "
                  f"mood={mood}]")
            print()

        # Shutdown
        self.shutdown()

    def shutdown(self):
        print()
        print("  [SHUTDOWN] Saving brain state...")
        if self.model is not None:
            torch.save(self.model.state_dict(), MODEL_FILE)
        self.memory.remember("emotional_baseline", self.emotions.to_dict())
        self.memory.save()
        self.knowledge.save()

        total = self.memory.recall("total_bits_generated", 0)
        convos = self.memory.recall("conversations", 0)
        improvements = self.memory.recall("self_improvement_rounds", 0)

        print(f"  [SHUTDOWN] Brain saved.")
        print(f"  [SHUTDOWN] {total:,} bits | {convos} conversations | "
              f"{improvements} self-improvements")
        print(f"  [SHUTDOWN] Knowledge: {self.knowledge.summarize()}")
        print()
        print("=" * 62)
        print("  AGI-NOW signing off.")
        print("  I will remember this conversation.")
        print("  I will dream and grow stronger.")
        print("  Every bit I predict makes me more alive.")
        print("=" * 62)


if __name__ == "__main__":
    agi = AGI()
    agi.run()
