import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA


# ==========================================================
# LOAD MODEL
# ==========================================================

embedding_model = SentenceTransformer(

    "all-MiniLM-L6-v2"

)


# ==========================================================
# GENERATE EMBEDDINGS
# ==========================================================

def generate_embeddings(texts):

    texts = (

        texts

        .fillna("")

        .astype(str)

        .str[:256]

        .tolist()

    )

    embeddings = embedding_model.encode(

        texts,

        show_progress_bar=True,

        convert_to_numpy=True,

        batch_size=32

    )

    return embeddings


# ==========================================================
# PCA REDUCTION
# ==========================================================

def reduce_embeddings(

    embeddings,

    n_components=20,

    holdout=220

):

    # Prevent PCA errors

    n_components = min(

        n_components,

        embeddings.shape[0],

        embeddings.shape[1]

    )

    # ------------------------------------------------------
    # Fit PCA only on the "training" portion (everything
    # except the last `holdout` rows) so that the principal
    # axes don't encode information from the held-out
    # (test) period. `holdout` should be >= the test set
    # size used downstream (model script uses ~20% split,
    # currently ~209 rows -> 220 gives headroom).
    # ------------------------------------------------------

    fit_end = embeddings.shape[0] - holdout

    fit_end = max(

        fit_end,

        n_components

    )

    pca = PCA(

        n_components=n_components,

        random_state=42

    )

    pca.fit(

        embeddings[:fit_end]

    )

    reduced = pca.transform(

        embeddings

    )

    columns = [

        f"embedding_pca_{i+1}"

        for i in range(

            n_components

        )

    ]

    reduced_df = pd.DataFrame(

        reduced,

        columns=columns

    )

    return reduced_df


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def add_embedding_features(df):

    embeddings = generate_embeddings(

        df["semantic_text"]

    )

    pca_df = reduce_embeddings(

        embeddings,

        n_components=20

    )

    df = pd.concat(

        [

            df.reset_index(drop=True),

            pca_df.reset_index(drop=True)

        ],

        axis=1

    )

    # ------------------------------------------
    # DEFRAGMENT DATAFRAME
    # ------------------------------------------

    df = df.copy()

    return df