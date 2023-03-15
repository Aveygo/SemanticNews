from sources import FEEDS
import feedparser, threading, queue, sqlite3, time, logging
from transformers import DistilBertTokenizerFast
from onnxruntime import InferenceSession
import torch, numpy as np, base64, zlib, time, os

class Downloader(threading.Thread):
    """
    Main downloader thread, downloads articles from RSS feeds and stores them in a database
    """
    def __init__(self, reoccuring:bool=False, threads:int=4):
        threading.Thread.__init__(self)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.logger.info("Starting downloader thread!")
        self.logger.info("Loading tokenizer...")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        if not os.path.exists("model.onnx"):
            self.logger.info("Model not found! Downloading and converting model...")
            from download_and_convert_bert import export_model
            export_model()

        self.logger.info("Loading model...")
        self.model = InferenceSession("model.onnx")

        self.logger.info("Loading database...")
        self.conn = sqlite3.connect('news.db', check_same_thread=False)
        self.c = self.conn.cursor()
        self.c.execute('''CREATE TABLE IF NOT EXISTS news
                        (news_id INTEGER PRIMARY KEY AUTOINCREMENT, title text, link text, summary text, published integer, source text, source_name text, media text, compvec text)''')

        self.download_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.threads = threads
        self.reoccuring = reoccuring

        self.downloading = False
        self.cache = {"time": 0, "kmeans": [], "headlines": []}
        self.lock = False
    
    def decode(self, x:str) -> np.ndarray:
        """
        Compvecs are stored as base64 encoded zlib compressed numpy arrays
        """
        try:
            return np.frombuffer(zlib.decompress(base64.urlsafe_b64decode(x)), dtype=np.float16)
        except:
            return np.ones(300) * 1000
    
    def kmeans(self, vectors:np.ndarray, num_clusters=5, max_iterations=50) -> list[np.ndarray]:
        """
        Find the center of each cluster of a multidimensional array
        """
        agents = []
        for agent in range(num_clusters):
            agents.append(vectors[np.random.randint(0, len(vectors))])

        for i in range(max_iterations):
            clusters = [[] for agent in range(num_clusters)]
            for vector in vectors:
                distances = []
                for agent in agents:
                    distances.append(np.linalg.norm(vector - agent))
                clusters[distances.index(min(distances))].append(vector)
            
            for agent in range(num_clusters):
                agents[agent] = np.mean(clusters[agent], axis=0)
                    
        return agents

    def parse_row(self, row:list[str]) -> dict:
        return {
            "news_id": row[0],
            "title": row[1],
            "link": row[2],
            "summary": row[3],
            "published": row[4],
            "source": row[5],
            "source_name": row[6],
            "media": row[7],
            "compvec": row[8]
        }
    
    def calculate_cache(self) -> dict:
        """
        Find clusters of articles and calculate a score for each article
        """
        # Only consider the last 1000 entries
        self.c.execute("SELECT * FROM news ORDER BY published DESC LIMIT 1000")
        articles = [self.parse_row(row) for row in self.c.fetchall()]
        decoded = [self.decode(article["compvec"]) for article in articles]
        
        # Calculate the cluster centers
        cache = self.cache
        cache["kmeans"] = self.kmeans(decoded)

        # Calculate score for each article
        for cluster_center in cache["kmeans"]:
            for i, article in enumerate(articles):
                article.setdefault("scores", [])
                article["scores"].append(1 / (1 + np.mean((decoded[i] - cluster_center) ** 2)))

        # Sort articles by score
        articles.sort(key=lambda x: max(x["scores"]), reverse=True)
        cache["headlines"] = articles
        cache["time"] = time.time()
        return cache

    def run(self):
        """
        Main loop, downloads articles and calculates headlines
        """
        while True:
            
            self.downloading = True
            self.download()
            self.cache = self.calculate_cache()
            if not self.reoccuring:
                break
            self.downloading = False
            time.sleep(60*60)
    
    def download(self):
        """
        Download articles from RSS feeds
        """
        self.logger.info("Starting article download...")
        
        # Start download threads
        for i in range(self.threads):
            t = threading.Thread(target=self.feed_thread, args=(i,))
            t.setDaemon(True)
            t.start()

        # Add all feeds to the download queue
        for feed in FEEDS:
            self.download_queue.put(feed)

        # Wait for all downloads to finish
        self.download_queue.join()
        self.logger.info("Finished article download!")

        # Add all results to the database, including calculating the compvec
        self.logger.info("Adding to database...")
        self.process_results()

    def sentence2compvec(self, sentence:str) -> str:
        """
        Converts a sentence to a encoded vector
        """
        marked_text = "[CLS] " + sentence.lower() + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = self.model.run(output_names=["last_hidden_state", "hidden_states", "attentions"], input_feed={"input_ids": tokens_tensor.numpy(), "attention_mask": segments_tensors.numpy()})
            hidden_states = outputs[1][0]
            token_vecs = hidden_states[-2]
            sentence_embedding = np.mean(token_vecs, axis=0)
        
        vector = sentence_embedding.astype(np.float16)
        return base64.urlsafe_b64encode(zlib.compress(vector)).decode("utf-8")

    def feed_thread(self, id):
        """
        Threaded, downloads articles that were added to the download queue and
        adds them to the results queue
        """
        while True:
            data = self.download_queue.get()
            url = data["url"]
            name = data["name"]
            self.results_queue.put((url, name, feedparser.parse(url)))
            self.download_queue.task_done()

    def process_results(self):
        """
        Adds all results from the results queue to the database
        and calculates the compvec for each article
        """
        num_entries = 0
        while self.results_queue.qsize() > 0:
            source, name, result = self.results_queue.get()
            for i, entry in enumerate(result.entries):

                title = entry.get("title", "")
                link = entry.get("link", "")
                
                # check if entry already exists (same link/title), check only last 1000 entries
                self.c.execute("SELECT * FROM news WHERE link=? OR title=? ORDER BY news_id DESC LIMIT 1000", (link, title))
                if self.c.fetchone() is None:

                    num_entries += 1

                    if num_entries % 100 == 0:
                        self.logger.info(f"Added {num_entries} entries to database")

                    media = entry.get("media_thumbnail", [{"url": ""}])[0]["url"]
                    if media == "":
                        media = entry.get("media_content", [{"url": ""}])[0]["url"]
                    
                    
                    summary = entry.get("summary", "")
                    if summary == "":
                        summary = entry.get("description", "")
                    
                    published = entry.get("published_parsed", "")
                    if published:
                        published = int(time.mktime(published))
                    else:
                        published = 0

                    if title:
                        compvec = self.sentence2compvec(title)
                    else:
                        compvec = ""
                
                    self.c.execute("INSERT INTO news (title, link, summary, published, source, source_name, media, compvec) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (title, link, summary, published, source, name, media, compvec))

            self.conn.commit()
            self.results_queue.task_done()


if __name__ == "__main__":
    start = time.time()
    d = Downloader()
    d.start()
    d.join()
    print(f"Finished in {time.time() - start:.2f} seconds")