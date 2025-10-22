from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_directional_model(feature_df, label_series):
    X = feature_df.values
    y = label_series.values

    if len(set(y)) < 2:
        return None, None

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    if len(X) > 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=max(1, int(len(X) * 0.2)),
            shuffle=False,
        )
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
    else:
        pipeline.fit(X, y)
        score = pipeline.score(X, y)

    return pipeline, score
