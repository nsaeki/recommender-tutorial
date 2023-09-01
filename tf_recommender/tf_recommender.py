# %%
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# %%
ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")
ratings = ratings.map(
    lambda x: {"movie_title": x["movie_title"], "user_id": x["user_id"]}
)
movies = movies.map(lambda x: x["movie_title"])

# %%
user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))
movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
movie_titles_vocabulary.adapt(movies)


# %%
class MovieLensModel(tfrs.Model):
    def __init__(
        self,
        user_model: tf.keras.Model,
        movie_model: tf.keras.Model,
        task: tfrs.tasks.Retrieval,
    ):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.task = task

    def compute_loss(self, features: dict[str, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])

        return self.task(user_embeddings, movie_embeddings)


# %%
user_model = tf.keras.Sequential(
    [
        user_ids_vocabulary,
        tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64),
    ]
)
movie_model = tf.keras.Sequential(
    [
        movie_titles_vocabulary,
        tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64),
    ]
)

task = tfrs.tasks.Retrieval(
    metrics=tfrs.metrics.FactorizedTopK(movies.batch(128).map(movie_model))
)

# %%
model = MovieLensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))
model.fit(ratings.batch(4096), epochs=3)

index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    movies.batch(100).map(lambda title: (title, model.movie_model(title)))
)
_, titles = index(np.array(["42"]))
print(f"Top 3 recommendations for user 42: {titles[0, :3]}")

