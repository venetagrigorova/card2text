import spacy
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----- visualize word vectors using t-SNE -----

# load spacy model
nlp = spacy.load("en_core_web_md")

# load OCR texts
with open("task-c-llama-3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# collect all entity texts
all_words = []
all_labels = []

for item in data:
    for key in ["Location", "Date", "Photographer", "Film", "Description"]:
        word = item.get(key, "")
        if word:
            for token in nlp(word):
                if token.has_vector:  # skip tokens with no vector
                    all_words.append(token)
                    all_labels.append(key)

# convert to vectors
vectors = [t.vector for t in all_words]

# reduce dimensions with t-SNE
tsne = TSNE(n_components=2, random_state=0, perplexity=5)
X_embedded = tsne.fit_transform(np.array(vectors))

# plot
plt.figure(figsize=(10, 8))
colors = {
    "Location": "red",
    "Date": "blue",
    "Photographer": "green",
    "Film": "purple",
    "Description": "orange"
}
for x, y, label in zip(X_embedded[:, 0], X_embedded[:, 1], all_labels):
    plt.scatter(x, y, c=colors[label], label=label, alpha=0.6, edgecolors='k')

# avoid duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title("t-SNE of Named Entity Word Vectors: llama-3")
plt.savefig("tsne-entities-llama-3.png", dpi=300, bbox_inches='tight')
plt.close()

# ----- compute and display word similarities -----

# get raw text for display
word_texts = [t.text for t in all_words]

# compute cosine similarities between vectors
sim_matrix = cosine_similarity(vectors)

# for each word, find top 3 most similar
with open("similar-words-llama-3.txt", "w", encoding="utf-8") as f:
    f.write("Word → Similar 1, Similar 2, Similar 3\n")
    f.write("=" * 50 + "\n")
    for i, word in enumerate(word_texts):
        sim_scores = sim_matrix[i]
        sorted_indices = np.argsort(sim_scores)[::-1][1:4]  # skip self
        similar_words = [word_texts[j] for j in sorted_indices]
        f.write(f"{word} → {', '.join(similar_words)}\n")

