import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt', quiet=True)

class TextRankSummarizer:
    def __init__(self, num_sentences=2):
        self.num_sentences = num_sentences

    def sentence_tokenize(self, text):
        return nltk.sent_tokenize(text)

    def build_similarity_matrix(self, sentences):
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(sentences)
        sim_matrix = cosine_similarity(tfidf, tfidf)
        return sim_matrix

    def pagerank(self, similarity_matrix, eps=0.0001, d=0.85):
        n = similarity_matrix.shape[0]
        scores = np.ones(n) / n

        while True:
            prev_scores = np.copy(scores)
            for i in range(n):
                s = 0
                for j in range(n):
                    if i != j:
                        s += similarity_matrix[j][i] * scores[j]
                scores[i] = (1 - d) + d * s

            if np.sum(np.fabs(scores - prev_scores)) <= eps:
                break

        return scores

    def summarize(self, text):
        sentences = self.sentence_tokenize(text)

        if len(sentences) <= self.num_sentences:
            return text  # already short

        sim_matrix = self.build_similarity_matrix(sentences)
        scores = self.pagerank(sim_matrix)

        ranked_sentences = [sentences[i] for i in np.argsort(scores)[-self.num_sentences:]]
        ranked_sentences.sort()  # keep original order

        return " ".join(ranked_sentences)


# Example usage

# text = """
# When the town’s old clock tower suddenly stopped ticking, everyone assumed it was just another mechanical failure—everyone except Liora, who had always sensed something alive beneath its chimes. Determined to prove her hunch, she slipped inside at dusk and discovered a tiny brass automaton slumped beside the gears, its eyes dim. After carefully winding its heart-shaped key, the automaton awakened and confessed it had kept time for centuries, growing tired but never daring to stop. Liora promised to help, and together they repaired the gears until the tower rang out again, this time with a new, warmer tone that echoed their unlikely partnership.

# """

# summarizer = TextRankSummarizer()
# summary = summarizer.summarize(text)
# print(summary)
