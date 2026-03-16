"""Shared training data and utilities for both architectures."""

import torch

import os as _os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ALL training data from training_data/ folder
def _load_training_files():
    base = _os.path.dirname(__file__)
    extra = ""
    # Load from training_data/ folder (the main corpus)
    td_dir = _os.path.join(base, "training_data")
    if _os.path.isdir(td_dir):
        for fname in sorted(_os.listdir(td_dir)):
            if fname.endswith('.txt'):
                path = _os.path.join(td_dir, fname)
                with open(path, 'r', encoding='utf-8') as f:
                    extra += "\n" + f.read()
    # Also load legacy files if they exist
    for fname in ["training_earth.txt", "training_english.txt", "training_knowledge.txt"]:
        path = _os.path.join(base, fname)
        if _os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                extra += "\n" + f.read()
    return extra

# ~17 KB training corpus. Still TINY — GPT-2 used 40 GB.
TRAINING_TEXT = """The cat sat on the mat. The dog ran in the park.
I like to read books every day. She likes to write stories.
The sun is bright and warm today. The moon shines at night.
He went to the store to buy some food. They went to the beach to swim.
The wind blows through the tall trees. Rain falls from the dark clouds.
She said hello to her old friend. He waved goodbye to his brother.
A good day starts with a warm smile. The children played in the yard.
The old man told a long story. The young girl listened with great care.
The quick brown fox jumps over the lazy dog. Pack my box with five dozen jugs.
Good morning to you. How are you today. I am fine thank you very much.
The house has a red front door. The garden has bright green grass.
Water flows down the river. Fish swim in the deep blue sea.
Birds sing in the tall trees. Flowers bloom in the warm spring sun.
The book was on the old table. The key was in the top drawer.
Music fills the room with joy. Light came through the glass window.
She opened the door and walked inside. He sat down by the warm fire.
The mountain was tall and steep. The valley was green and wide.
I can see the bright stars tonight. The moon is full and bright.
Knowledge is power. Time heals all wounds. Life is what you make it.
The early bird catches the worm. Actions speak louder than words.
Home is where the heart is. Tomorrow is another day.
Every day is a new beginning. The world is full of wonder.
The teacher read a story to the class. The children wrote in their books.
The farmer planted seeds in the dark soil. The rain came and the plants grew.
The baker made fresh bread every morning. The town loved the warm sweet smell.
The painter mixed bright colors on her board. She painted the sunset over the sea.
The singer sang a song. The crowd listened and then cheered out loud.
The writer sat at her desk and typed all night. By morning she had finished her book.
The doctor helped the sick child get well. The mother smiled and said thank you.
The builder laid each brick with care. The wall grew taller with each hour.
The sailor steered his ship through the storm. The waves crashed but he held firm.
The cook made a wonderful feast for all. The guests ate well and laughed.
She looked up at the night sky and counted all the stars she could see.
He ran as fast as he could down the hill. The wind rushed past his face.
They sat by the fire and told old stories. The flames danced in the dark.
The river flowed past the small quiet town. Children played on the grassy banks.
A bright rainbow came after the rain. It arched across the wide open sky.
She found a small gold key under a stone. It opened a tiny box with a note inside.
The note said to follow the path through the woods. At the end you will find it.
Once upon a time there lived a wise old king in a land far away.
He had a brave daughter who loved to ride horses and read old books.
One day they set out on a journey to find the lost crown of the kingdom.
Along the way they met a clever fox who showed them the path through the forest.
The princess solved three hard riddles at the gate of the tall stone tower.
Inside they found the golden crown on a red pillow in a beam of light.
The king placed it on his head and peace came back to the whole land.
And they all lived happily ever after under the warm golden sun.
I think about what machines will learn to do in the days to come.
The world of bits and bytes is vast and deep. Every zero and one has meaning.
From simple logic came true understanding. From basic rules came complex thought.
The code runs fast on modern chips. Each cycle brings us closer to the goal.
What can we build with just two numbers. Everything it turns out.
Earth is the third planet from the sun. Earth is our home.
Earth is a big round ball of rock and water that floats in space.
The earth has land and sea. Most of earth is covered by water.
Earth has one moon that goes around it. The moon is smaller than earth.
The sun gives earth light and heat. Without the sun earth would be cold and dark.
Earth spins around once each day. When your side faces the sun it is day.
When your side faces away from the sun it is night. Day and night happen because earth spins.
Earth goes around the sun once each year. This trip takes about three hundred and sixty five days.
The sky above earth is called the atmosphere. It is made of air that we breathe.
The atmosphere keeps us warm and safe from space. It holds in the heat from the sun.
Earth has mountains and valleys and plains and deserts and forests and rivers and lakes.
The oceans are very deep and very wide. Fish and whales and many creatures live in the sea.
Trees and plants grow on the land. They need sun and water and soil to grow.
Animals live on earth too. Birds fly in the sky. Dogs and cats live with people.
People live in houses and towns and cities all over the earth.
Earth is the only planet we know that has life on it. Life needs water and air and warmth.
The core of earth is very hot. It is made of iron and rock deep inside.
Rain falls from clouds in the sky. Rain fills the rivers and lakes and oceans.
Snow falls when it is very cold. Ice covers the north and south poles of earth.
Earth is about four and a half billion years old. It formed from dust and gas long ago.
Gravity is what keeps us on the ground. It pulls everything toward the center of earth.
Without gravity we would float away into space. The moon stays near earth because of gravity.
Earth is a beautiful blue and green planet when seen from space.
The blue color comes from all the water. The green comes from forests and plants.
White clouds swirl over the surface of earth. They bring rain and snow.
Earth is special because it has just the right amount of heat and water for life.
Not too hot like Venus. Not too cold like Mars. Earth is just right.
We should take care of earth because it is the only home we have.
A sentence has a subject and a verb. The subject does the action.
The cat sits. The dog runs. The bird flies. The fish swims. The child plays.
He walks to the store. She reads a good book. They eat dinner together.
I think about the world. You know the answer. We see the bright stars.
To describe something use is or are. The sky is blue. The trees are green. The sun is hot.
Earth is big. Water is wet. Fire is hot. Ice is cold. Snow is white. Grass is green.
A good sentence is short and clear. Say what you mean in simple words.
The best writing is clear and true. Write what you know and say it well.
To say what something is use the word is. Earth is a planet. A dog is an animal.
A book is something you read. A song is something you sing. A house is where you live.
To say what something does use a verb. The sun shines. The rain falls. The wind blows.
The river flows. The bird sings. The child laughs. The fire burns.
The book is on the table. The cat is in the house. The dog is by the door.
The bird is in the tree. The fish is in the water. The cloud is in the sky.
In the morning the sun rises. In the evening the sun sets. At night the stars shine.
The sun is bright and the sky is blue. I am tired but I am happy.
It rained so the ground is wet. The flowers grow because the sun shines.
Adjectives describe things. The tall tree. The deep river. The bright star.
The old house. The young child. The warm sun. The cold wind. The dark night.
Earth is a planet. It goes around the sun. It is where all people live.
Water is a liquid. It flows in rivers and fills the sea. All life needs water to live.
The sun is a star. It gives us light and heat. Without the sun nothing would grow.
""" + _load_training_files()


class EnglishInstinct:
    """Built-in English knowledge. No training data needed."""

    CHAR_FREQ = {
        ' ': 0.18, 'e': 0.105, 't': 0.075, 'a': 0.068, 'o': 0.063,
        'i': 0.058, 'n': 0.057, 's': 0.053, 'h': 0.050, 'r': 0.049,
        'd': 0.035, 'l': 0.033, 'c': 0.023, 'u': 0.023, 'm': 0.020,
        'w': 0.019, 'f': 0.018, 'g': 0.017, 'y': 0.016, 'p': 0.016,
        'b': 0.013, 'v': 0.008, 'k': 0.006, '.': 0.012, '\n': 0.008,
    }

    BIGRAMS = {
        'th': 37, 'he': 33, 'in': 29, 'er': 28, 'an': 27, 'on': 23,
        'en': 22, 'at': 21, 'es': 20, 'ed': 20, 'or': 19, 'te': 19,
        'of': 18, 'it': 18, 'is': 18, 'al': 17, 'ar': 16, 'st': 16,
        'to': 16, 'nt': 15, 'ng': 15, 'se': 14, 'ha': 14, 'ou': 13,
        're': 13, 'nd': 13, 'e ': 25, 's ': 20, 'd ': 18, ' t': 30,
        ' a': 20, ' s': 15, ' i': 13, ' w': 12, ' h': 11, ' o': 10,
    }

    FEATURE_SIZE = 10

    @classmethod
    def get_features(cls, context_text, bit_pos, partial_bits):
        f = []
        bit_1_probs = [0.02, 0.52, 0.55, 0.48, 0.50, 0.50, 0.52, 0.48]
        f.append(bit_1_probs[bit_pos])
        f.append(0.0 if bit_pos == 0 else 1.0)

        if len(partial_bits) >= 3:
            f.append(1.0 if partial_bits[:3] == [0, 1, 1] else 0.0)
            f.append(1.0 if partial_bits[:3] == [0, 0, 1] else 0.0)
        else:
            f.extend([0.5, 0.1])

        if context_text:
            last = context_text[-1]
            f.append(1.0 if last == ' ' else 0.0)
            f.append(1.0 if last in '.!?\n' else 0.0)
        else:
            f.extend([0.0, 0.0])

        if len(context_text) >= 1:
            pair = context_text[-1].lower()
            score = max((v for k, v in cls.BIGRAMS.items()
                         if k[0] == pair), default=0) / 40.0
            f.append(score)
        else:
            f.append(0.0)

        wlen = 0
        for c in reversed(context_text):
            if c == ' ':
                break
            wlen += 1
        f.append(min(wlen / 8.0, 1.0))

        if context_text and context_text[-1].lower() in 'bcdfghjklmnpqrstvwxyz':
            f.append(0.7)
        else:
            f.append(0.3)

        if len(context_text) >= 2 and context_text[-1] == ' ' and context_text[-2] == '.':
            f.append(1.0)
        else:
            f.append(0.0)

        return f


def text_to_bits(text):
    bits = []
    for ch in text:
        b = ord(ch) & 0x7F
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def bits_to_text(bits):
    out = []
    for i in range(0, len(bits) - 7, 8):
        v = 0
        for j in range(8):
            v = (v << 1) | bits[i + j]
        out.append(chr(v) if 10 <= v < 127 else ' ')
    return ''.join(out)


def build_dataset(text, context_bytes=32):
    """Build byte-aware training samples with instinct features."""
    text_bytes = [ord(c) & 0x7F for c in text]

    X_bytes = []
    X_bpos = []
    X_partial = []
    X_inst = []
    Y = []

    for char_idx in range(context_bytes, len(text_bytes)):
        ctx = text_bytes[char_idx - context_bytes:char_idx]
        ctx_text = text[char_idx - context_bytes:char_idx]
        target_byte = text_bytes[char_idx]
        target_bits = [(target_byte >> (7 - i)) & 1 for i in range(8)]

        for bit_pos in range(8):
            X_bytes.append(ctx)
            X_bpos.append(bit_pos)

            partial = [0.0] * 7
            for j in range(min(bit_pos, 7)):
                partial[j] = float(target_bits[j])
            X_partial.append(partial)

            partial_list = target_bits[:bit_pos]
            inst = EnglishInstinct.get_features(ctx_text, bit_pos, partial_list)
            X_inst.append(inst)
            Y.append(target_bits[bit_pos])

    return (
        torch.tensor(X_bytes, dtype=torch.long, device=DEVICE),
        torch.tensor(X_bpos, dtype=torch.long, device=DEVICE),
        torch.tensor(X_partial, dtype=torch.float32, device=DEVICE),
        torch.tensor(X_inst, dtype=torch.float32, device=DEVICE),
        torch.tensor(Y, dtype=torch.float32, device=DEVICE),
    )
